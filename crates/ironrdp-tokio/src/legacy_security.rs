//! Transport adapter applying Standard RDP Security (RC4) to whole frames of an underlying byte stream.

use core::pin::Pin;
use core::task::{Context, Poll, ready};
use std::io;

use ironrdp_connector::legacy_security::LegacySecurityHandle;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};

const READ_CHUNK_SIZE: usize = 8 * 1024;

/// Wraps a byte stream so that the layers above see plaintext frames while the wire carries
/// frames protected by the [`LegacySecurityHandle`] codec.
///
/// While the handle is not armed the stream is byte-for-byte transparent and needs no frame
/// alignment. Once armed, bytes are cut into whole TPKT / fast-path frames (a frame may span
/// several reads or writes, and one read may hold several frames) before the codec sees them.
///
/// Arming is decided per frame at the moment it is processed: bytes are never decrypted
/// retroactively. The caller must therefore arm at a frame boundary, i.e. before the first
/// protected frame is read from the stream or written to it.
pub struct LegacySecurityStream<S> {
    inner: S,
    handle: LegacySecurityHandle,
    /// Bytes read from `inner` that do not yet form a whole frame.
    raw_in: Vec<u8>,
    /// Decoded bytes ready to be served to the reader, from `plain_pos`.
    plain_in: Vec<u8>,
    plain_pos: usize,
    eof: bool,
    /// Bytes accepted from the writer that do not yet form a whole frame.
    plain_out: Vec<u8>,
    /// Encoded bytes waiting to be written to `inner`, from `wire_pos`.
    wire_out: Vec<u8>,
    wire_pos: usize,
}

impl<S> core::fmt::Debug for LegacySecurityStream<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LegacySecurityStream")
            .field("armed", &self.handle.is_armed())
            .field("raw_in_len", &self.raw_in.len())
            .field("plain_in_len", &self.plain_in.len())
            .field("plain_out_len", &self.plain_out.len())
            .field("wire_out_len", &self.wire_out.len())
            .finish_non_exhaustive()
    }
}

impl<S> LegacySecurityStream<S> {
    pub fn new(inner: S, handle: LegacySecurityHandle) -> Self {
        Self {
            inner,
            handle,
            raw_in: Vec::new(),
            plain_in: Vec::new(),
            plain_pos: 0,
            eof: false,
            plain_out: Vec::new(),
            wire_out: Vec::new(),
            wire_pos: 0,
        }
    }

    pub fn get_ref(&self) -> &S {
        &self.inner
    }

    pub fn into_inner(self) -> S {
        self.inner
    }

    /// Moves whole frames (everything, while unarmed) from `src` into `dst` through `transform`.
    fn convert(
        handle: &LegacySecurityHandle,
        src: &mut Vec<u8>,
        dst: &mut Vec<u8>,
        transform: fn(&LegacySecurityHandle, &[u8]) -> ironrdp_connector::ConnectorResult<Vec<u8>>,
    ) -> io::Result<()> {
        if !handle.is_armed() {
            dst.append(src);
            return Ok(());
        }

        loop {
            let info = ironrdp_pdu::find_size(src).map_err(invalid_data)?;
            let Some(info) = info else {
                return Ok(());
            };
            if info.length == 0 {
                return Err(invalid_data("zero-length frame"));
            }
            if src.len() < info.length {
                return Ok(());
            }
            let converted = transform(handle, &src[..info.length]).map_err(invalid_data)?;
            dst.extend_from_slice(&converted);
            src.drain(..info.length);
        }
    }

    fn convert_inbound(&mut self) -> io::Result<()> {
        if self.plain_pos == self.plain_in.len() {
            self.plain_in.clear();
            self.plain_pos = 0;
        }
        Self::convert(
            &self.handle,
            &mut self.raw_in,
            &mut self.plain_in,
            LegacySecurityHandle::decrypt_inbound,
        )
    }

    fn convert_outbound(&mut self) -> io::Result<()> {
        Self::convert(
            &self.handle,
            &mut self.plain_out,
            &mut self.wire_out,
            LegacySecurityHandle::encrypt_outbound,
        )
    }
}

fn invalid_data<E>(error: E) -> io::Error
where
    E: Into<Box<dyn core::error::Error + Send + Sync>>,
{
    io::Error::new(io::ErrorKind::InvalidData, error)
}

impl<S: AsyncWrite + Unpin> LegacySecurityStream<S> {
    /// Writes pending encoded bytes to `inner`; `Ready` once all of them are written.
    fn poll_drain(&mut self, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        while self.wire_pos < self.wire_out.len() {
            let n = ready!(Pin::new(&mut self.inner).poll_write(cx, &self.wire_out[self.wire_pos..]))?;
            if n == 0 {
                return Poll::Ready(Err(io::ErrorKind::WriteZero.into()));
            }
            self.wire_pos += n;
        }
        self.wire_out.clear();
        self.wire_pos = 0;
        Poll::Ready(Ok(()))
    }
}

impl<S: AsyncRead + Unpin> AsyncRead for LegacySecurityStream<S> {
    fn poll_read(mut self: Pin<&mut Self>, cx: &mut Context<'_>, buf: &mut ReadBuf<'_>) -> Poll<io::Result<()>> {
        let this = &mut *self;

        if buf.remaining() == 0 {
            return Poll::Ready(Ok(()));
        }

        loop {
            if this.plain_pos < this.plain_in.len() {
                let available = &this.plain_in[this.plain_pos..];
                let n = available.len().min(buf.remaining());
                buf.put_slice(&available[..n]);
                this.plain_pos += n;
                return Poll::Ready(Ok(()));
            }

            this.convert_inbound()?;
            if this.plain_pos < this.plain_in.len() {
                continue;
            }

            if this.eof {
                if this.raw_in.is_empty() {
                    return Poll::Ready(Ok(()));
                }
                return Poll::Ready(Err(io::ErrorKind::UnexpectedEof.into()));
            }

            let mut chunk = [0; READ_CHUNK_SIZE];
            let mut chunk_buf = ReadBuf::new(&mut chunk);
            ready!(Pin::new(&mut this.inner).poll_read(cx, &mut chunk_buf))?;
            if chunk_buf.filled().is_empty() {
                this.eof = true;
            } else {
                this.raw_in.extend_from_slice(chunk_buf.filled());
            }
        }
    }
}

impl<S: AsyncWrite + Unpin> AsyncWrite for LegacySecurityStream<S> {
    fn poll_write(mut self: Pin<&mut Self>, cx: &mut Context<'_>, buf: &[u8]) -> Poll<io::Result<usize>> {
        let this = &mut *self;

        // Backpressure: do not accept more while earlier frames are still waiting for the inner stream.
        ready!(this.poll_drain(cx))?;

        this.plain_out.extend_from_slice(buf);
        this.convert_outbound()?;
        // Best effort: the bytes are accepted either way and `poll_flush` drains the rest. Returning
        // an error here would make a retrying caller duplicate the frame and advance RC4 twice; a
        // persistent error resurfaces on the next `poll_write` or `poll_flush`.
        let _ = this.poll_drain(cx);

        Poll::Ready(Ok(buf.len()))
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = &mut *self;

        this.convert_outbound()?;
        ready!(this.poll_drain(cx))?;
        Pin::new(&mut this.inner).poll_flush(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        ready!(self.as_mut().poll_flush(cx))?;
        // A partial plaintext frame would be dropped silently; mirror `poll_read`'s `UnexpectedEof`.
        if !self.plain_out.is_empty() {
            return Poll::Ready(Err(io::ErrorKind::UnexpectedEof.into()));
        }
        Pin::new(&mut self.inner).poll_shutdown(cx)
    }
}

#[cfg(test)]
mod tests {
    use core::task::Waker;

    use ironrdp_connector::legacy_security::{LegacySecurityCodec, SessionKeys, mac};
    use ironrdp_core::{decode, encode_vec};
    use ironrdp_pdu::gcc::EncryptionMethod;
    use ironrdp_pdu::mcs::{SendDataIndication, SendDataRequest};
    use ironrdp_pdu::x224::X224;
    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _, DuplexStream, duplex};

    use super::*;

    const IO: u16 = 1003;
    const SIGNATURE_SIZE: usize = 8;
    const EXCHANGE_PKT: u16 = 0x0001;
    const INFO_PKT: u16 = 0x0040;
    const ENCRYPT: u16 = 0x0008;

    fn keys() -> SessionKeys {
        SessionKeys::derive(&[0x11; 32], &[0x22; 32], EncryptionMethod::BIT_128).unwrap()
    }

    /// Server-side view of the same keys: encrypt and decrypt swapped.
    fn server_keys() -> SessionKeys {
        let SessionKeys {
            mac_key,
            encrypt,
            decrypt,
        } = keys();
        SessionKeys {
            mac_key,
            encrypt: decrypt,
            decrypt: encrypt,
        }
    }

    fn armed_handle() -> LegacySecurityHandle {
        let handle = LegacySecurityHandle::default();
        handle.arm(LegacySecurityCodec::new(keys(), IO, None));
        handle
    }

    fn header(flags: u16) -> Vec<u8> {
        [flags.to_le_bytes().as_slice(), &[0, 0]].concat()
    }

    fn request(user_data: &[u8]) -> Vec<u8> {
        encode_vec(&X224(SendDataRequest {
            initiator_id: 1002,
            channel_id: IO,
            user_data: user_data.into(),
        }))
        .unwrap()
    }

    fn indication(user_data: &[u8]) -> Vec<u8> {
        encode_vec(&X224(SendDataIndication {
            initiator_id: 1002,
            channel_id: IO,
            user_data: user_data.into(),
        }))
        .unwrap()
    }

    fn exchange_frame() -> Vec<u8> {
        request(&[header(EXCHANGE_PKT), vec![7; 20]].concat())
    }

    fn info_frame(payload: &[u8]) -> Vec<u8> {
        request(&[header(INFO_PKT), payload.to_vec()].concat())
    }

    /// What a server does to an encrypted client frame: strip the header, decrypt, verify the MAC.
    fn server_open_request(keys: &mut SessionKeys, frame: &[u8]) -> Vec<u8> {
        let X224(request) = decode::<X224<SendDataRequest<'_>>>(frame).unwrap();
        let (signature, ciphertext) = request.user_data[4..].split_at(SIGNATURE_SIZE);
        let mut plaintext = ciphertext.to_vec();
        keys.decrypt.process(&mut plaintext);
        assert_eq!(mac(&keys.mac_key, &plaintext), signature);
        plaintext
    }

    /// What a server does to a payload: MAC, RC4, encrypted security header.
    fn server_seal_indication(keys: &mut SessionKeys, payload: &[u8]) -> Vec<u8> {
        let signature = mac(&keys.mac_key, payload);
        let mut body = payload.to_vec();
        keys.encrypt.process(&mut body);
        indication(&[header(ENCRYPT), signature.to_vec(), body].concat())
    }

    fn pair(handle: LegacySecurityHandle) -> (LegacySecurityStream<DuplexStream>, DuplexStream) {
        let (client, server) = duplex(64 * 1024);
        (LegacySecurityStream::new(client, handle), server)
    }

    async fn read_exactly(stream: &mut (impl AsyncRead + Unpin), len: usize) -> Vec<u8> {
        let mut out = vec![0; len];
        stream.read_exact(&mut out).await.unwrap();
        out
    }

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn debug_output_shows_lengths_never_buffer_contents() {
        let (client, _server) = duplex(64);
        let mut stream = LegacySecurityStream::new(client, armed_handle());
        stream.raw_in = vec![0xA1, 0xA2, 0xA3];
        stream.plain_in = b"hunter2-secret".to_vec();
        stream.plain_out = b"password-bytes".to_vec();
        stream.wire_out = vec![0xB1, 0xB2];

        let dump = format!("{stream:?}");

        assert!(dump.contains("armed: true"), "{dump}");
        assert!(dump.contains("plain_in_len: 14"), "{dump}");
        for leaked in ["hunter2", "password", "161", "177", "104", "[", "]"] {
            assert!(!dump.contains(leaked), "{leaked} leaked in {dump}");
        }
    }

    #[test]
    fn stream_is_send_and_sync() {
        assert_send_sync::<LegacySecurityStream<DuplexStream>>();
    }

    #[tokio::test]
    async fn unarmed_is_transparent() {
        let (mut stream, mut server) = pair(LegacySecurityHandle::default());
        let bytes = [3, 0, 0, 40, 1, 2, 3];

        stream.write_all(&bytes).await.unwrap();
        stream.flush().await.unwrap();
        assert_eq!(read_exactly(&mut server, bytes.len()).await, bytes);

        server.write_all(&bytes).await.unwrap();
        assert_eq!(read_exactly(&mut stream, bytes.len()).await, bytes);
    }

    #[tokio::test]
    async fn armed_round_trip_through_duplex() {
        let (mut stream, mut server) = pair(armed_handle());
        let mut server_keys = server_keys();

        stream.write_all(&exchange_frame()).await.unwrap();
        stream.write_all(&info_frame(&[0xAB; 50])).await.unwrap();
        stream.flush().await.unwrap();

        assert_eq!(
            read_exactly(&mut server, exchange_frame().len()).await,
            exchange_frame()
        );
        let wire = read_exactly(&mut server, info_frame(&[0xAB; 50]).len() + SIGNATURE_SIZE).await;
        assert_eq!(server_open_request(&mut server_keys, &wire), [0xAB; 50]);

        let sealed = server_seal_indication(&mut server_keys, &[0x42; 30]);
        server.write_all(&sealed).await.unwrap();
        let plain = indication(&[0x42; 30]);
        assert_eq!(read_exactly(&mut stream, plain.len()).await, plain);
    }

    #[tokio::test]
    async fn frame_split_across_writes() {
        let (mut stream, mut server) = pair(armed_handle());
        let mut server_keys = server_keys();
        stream.write_all(&exchange_frame()).await.unwrap();
        stream.flush().await.unwrap();
        read_exactly(&mut server, exchange_frame().len()).await;

        let frame = info_frame(&[0xCD; 40]);
        let (first, rest) = frame.split_at(2);
        let (second, third) = rest.split_at(10);

        for chunk in [first, second] {
            stream.write_all(chunk).await.unwrap();
            stream.flush().await.unwrap();
        }
        let mut probe = [0; 1];
        let mut probe_buf = ReadBuf::new(&mut probe);
        let mut cx = Context::from_waker(Waker::noop());
        assert!(Pin::new(&mut server).poll_read(&mut cx, &mut probe_buf).is_pending());

        stream.write_all(third).await.unwrap();
        stream.flush().await.unwrap();

        let wire = read_exactly(&mut server, frame.len() + SIGNATURE_SIZE).await;
        assert_eq!(server_open_request(&mut server_keys, &wire), [0xCD; 40]);
    }

    #[tokio::test]
    async fn two_frames_in_one_read() {
        let (mut stream, mut server) = pair(armed_handle());
        let mut server_keys = server_keys();

        let both = [
            server_seal_indication(&mut server_keys, &[1; 10]),
            server_seal_indication(&mut server_keys, &[2; 20]),
        ]
        .concat();
        server.write_all(&both).await.unwrap();

        let expected = [indication(&[1; 10]), indication(&[2; 20])].concat();
        assert_eq!(read_exactly(&mut stream, expected.len()).await, expected);
    }

    #[tokio::test]
    async fn inbound_frame_split_across_reads() {
        let (mut stream, mut server) = pair(armed_handle());
        let mut server_keys = server_keys();
        let sealed = server_seal_indication(&mut server_keys, &[9; 25]);
        let (head, tail) = sealed.split_at(5);

        server.write_all(head).await.unwrap();
        let reader = tokio::spawn(async move {
            let plain = indication(&[9; 25]);
            (read_exactly(&mut stream, plain.len()).await, plain)
        });
        tokio::task::yield_now().await;
        server.write_all(tail).await.unwrap();

        let (got, expected) = reader.await.unwrap();
        assert_eq!(got, expected);
    }

    #[tokio::test]
    async fn codec_error_surfaces_as_invalid_data() {
        let (mut stream, mut server) = pair(armed_handle());
        let mut server_keys = server_keys();
        let mut sealed = server_seal_indication(&mut server_keys, &[5; 16]);
        let last = sealed.len() - 1;
        sealed[last] ^= 0xFF;
        server.write_all(&sealed).await.unwrap();

        let error = stream.read(&mut [0; 64]).await.unwrap_err();

        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }

    /// Fails the next write with `Interrupted`, then behaves like the wrapped stream.
    struct FailNextWrite<S> {
        inner: S,
        fail_next: bool,
    }

    impl<S: AsyncWrite + Unpin> AsyncWrite for FailNextWrite<S> {
        fn poll_write(mut self: Pin<&mut Self>, cx: &mut Context<'_>, buf: &[u8]) -> Poll<io::Result<usize>> {
            if core::mem::take(&mut self.fail_next) {
                return Poll::Ready(Err(io::ErrorKind::Interrupted.into()));
            }
            Pin::new(&mut self.inner).poll_write(cx, buf)
        }

        fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            Pin::new(&mut self.inner).poll_flush(cx)
        }

        fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            Pin::new(&mut self.inner).poll_shutdown(cx)
        }
    }

    #[tokio::test]
    async fn retried_write_after_inner_error_does_not_duplicate_the_frame() {
        let (client, mut server) = duplex(64 * 1024);
        let mut stream = LegacySecurityStream::new(
            FailNextWrite {
                inner: client,
                fail_next: false,
            },
            armed_handle(),
        );
        let mut server_keys = server_keys();
        stream.write_all(&exchange_frame()).await.unwrap();
        stream.flush().await.unwrap();
        read_exactly(&mut server, exchange_frame().len()).await;

        let frame = info_frame(&[0xEE; 30]);
        stream.inner.fail_next = true;
        stream.write_all(&frame).await.unwrap();
        stream.flush().await.unwrap();

        let wire = read_exactly(&mut server, frame.len() + SIGNATURE_SIZE).await;
        assert_eq!(server_open_request(&mut server_keys, &wire), [0xEE; 30]);
        let mut probe = [0; 1];
        let mut probe_buf = ReadBuf::new(&mut probe);
        let mut cx = Context::from_waker(Waker::noop());
        assert!(
            Pin::new(&mut server).poll_read(&mut cx, &mut probe_buf).is_pending(),
            "the frame was written twice"
        );
    }

    #[tokio::test]
    async fn zero_capacity_read_is_ready_without_reading_the_inner_stream() {
        let (mut stream, _server) = pair(armed_handle());
        let mut empty: [u8; 0] = [];
        let mut buf = ReadBuf::new(&mut empty);
        let mut cx = Context::from_waker(Waker::noop());

        let poll = Pin::new(&mut stream).poll_read(&mut cx, &mut buf);

        assert!(matches!(poll, Poll::Ready(Ok(()))), "{poll:?}");
    }

    #[tokio::test]
    async fn shutdown_with_partial_frame_is_an_error() {
        let (mut stream, _server) = pair(armed_handle());
        let frame = exchange_frame();
        stream.write_all(&frame[..5]).await.unwrap();

        let error = stream.shutdown().await.unwrap_err();

        assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
    }
}
