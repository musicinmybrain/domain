//! XFR request handling middleware.

use core::future::{ready, Future, Ready};
use core::marker::PhantomData;
use core::ops::ControlFlow;
use core::pin::Pin;

use std::boxed::Box;
use std::sync::Arc;

use bytes::Bytes;
use futures::stream::{once, Once, Stream};
use octseq::Octets;
use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};
use tokio::sync::Semaphore;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{error, info, trace, warn};

use crate::base::iana::{Opcode, OptRcode, Rcode};
use crate::base::message_builder::{
    AdditionalBuilder, AnswerBuilder, PushError,
};
use crate::base::net::IpAddr;
use crate::base::record::ComposeRecord;
use crate::base::wire::Composer;
use crate::base::{
    Message, Name, ParsedName, Question, Rtype, Serial, StreamTarget, ToName,
};
use crate::net::server::message::{Request, TransportSpecificContext};
use crate::net::server::service::{
    CallResult, Service, ServiceFeedback, ServiceResult,
};
use crate::net::server::util::{mk_builder_for_target, mk_error_response};
use crate::rdata::{Soa, ZoneRecordData};
use crate::tsig::{KeyName, KeyStore};
use crate::zonecatalog::catalog::{
    Catalog, CatalogZone, CompatibilityMode, TsigKey, XfrSettings,
    XfrStrategy, ZoneInfo, ZoneType,
};
use crate::zonetree::error::OutOfZone;
use crate::zonetree::{
    Answer, AnswerContent, ReadableZone, SharedRrset, StoredName,
};

use super::stream::MiddlewareStream;

//------------ XfrMapStream --------------------------------------------------

type XfrResultStream<StreamItem> = UnboundedReceiverStream<StreamItem>;

//------------ XfrMiddlewareStream -------------------------------------------

type XfrMiddlewareStream<Future, Stream, StreamItem> = MiddlewareStream<
    Future,
    Stream,
    Once<Ready<StreamItem>>,
    XfrResultStream<StreamItem>,
    StreamItem,
>;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum XfrMode {
    AxfrAndIxfr,
    AxfrOnly,
}

//------------ XfrMiddlewareSvc ----------------------------------------------

/// A [`MiddlewareProcessor`] for responding to XFR requests.
///
/// Standards covered by ths implementation:
///
/// | RFC    | Status  |
/// |--------|---------|
/// | [1034] | TBD     |
/// | [1035] | TBD     |
/// | [1995] | TBD     |
/// | [5936] | TBD     |
///
/// [`MiddlewareProcessor`]:
///     crate::net::server::middleware::processor::MiddlewareProcessor
/// [1034]: https://datatracker.ietf.org/doc/html/rfc1034
/// [1035]: https://datatracker.ietf.org/doc/html/rfc1035
/// [1995]: https://datatracker.ietf.org/doc/html/rfc1995
/// [5936]: https://datatracker.ietf.org/doc/html/rfc5936
#[derive(Clone, Debug)]
pub struct XfrMiddlewareSvc<RequestOctets, NextSvc, KS> {
    next_svc: NextSvc,

    catalog: Arc<Catalog<KS>>,

    zone_walking_semaphore: Arc<Semaphore>,

    batcher_semaphore: Arc<Semaphore>,

    xfr_mode: XfrMode,

    _phantom: PhantomData<RequestOctets>,
}

impl<RequestOctets, NextSvc, KS>
    XfrMiddlewareSvc<RequestOctets, NextSvc, KS>
{
    /// Creates an empty processor instance.
    ///
    /// The processor will not respond to XFR requests until you add at least
    /// one zone to it.
    // TODO: Move extra arguments into a Config object.
    #[must_use]
    pub fn new(
        next_svc: NextSvc,
        catalog: Arc<Catalog<KS>>,
        max_concurrency: usize,
        xfr_mode: XfrMode,
    ) -> Self {
        let zone_walking_semaphore =
            Arc::new(Semaphore::new(max_concurrency));
        let batcher_semaphore = Arc::new(Semaphore::new(max_concurrency));

        Self {
            next_svc,
            catalog,
            zone_walking_semaphore,
            batcher_semaphore,
            xfr_mode,
            _phantom: PhantomData,
        }
    }
}

impl<RequestOctets, NextSvc, KS> XfrMiddlewareSvc<RequestOctets, NextSvc, KS>
where
    RequestOctets: Octets + Send + Sync + 'static + Unpin,
    for<'a> <RequestOctets as octseq::Octets>::Range<'a>: Send + Sync,
    NextSvc: Service<RequestOctets, ()> + Clone + Send + Sync + 'static,
    NextSvc::Future: Send + Sync + Unpin,
    NextSvc::Target: Composer + Default + Send + Sync,
    NextSvc::Stream: Send + Sync,
    KS: KeyStore + Clone + Send + Sync + 'static,
    KS::Key: Send + Sync + 'static,
{
    async fn preprocess<T>(
        zone_walking_semaphore: Arc<Semaphore>,
        batcher_semaphore: Arc<Semaphore>,
        req: &Request<RequestOctets, T>,
        catalog: Arc<Catalog<KS>>,
        xfr_mode: XfrMode,
        get_key_for_req: fn(&Request<RequestOctets, T>) -> Option<&KeyName>,
    ) -> ControlFlow<
        XfrMiddlewareStream<
            NextSvc::Future,
            NextSvc::Stream,
            <NextSvc::Stream as Stream>::Item,
        >,
    > {
        let msg = req.message();

        let Some(q) = Self::get_relevant_question(msg) else {
            return ControlFlow::Continue(());
        };

        let qname: Name<Bytes> = q.qname().to_name();

        // Find the zone
        let zones = catalog.zones();
        let Some(zone) = zones.get_zone(&qname, q.qclass()) else {
            // https://datatracker.ietf.org/doc/html/rfc5936#section-2.2.1
            // 2.2.1 Header Values
            //   "If a server is not authoritative for the queried zone, the
            //    server SHOULD set the value to NotAuth(9)"
            warn!(
                "{} for {qname} from {} refused: unknown zone",
                q.qtype(),
                req.client_addr()
            );

            // Note: This may not be strictly true, we may be authoritative
            // for the zone but not willing to transfer it, but we can't know
            // here which is the case.
            return ControlFlow::Break(Self::to_stream(mk_error_response(
                msg,
                OptRcode::NOTAUTH,
            )));
        };

        // Only provide XFR if allowed.
        let cat_zone = zone
            .as_ref()
            .as_any()
            .downcast_ref::<CatalogZone>()
            .unwrap();

        let (xfr_settings, tsig_key) = Self::settings_for_client(
            req.client_addr().ip(),
            &qname,
            q.qtype(),
            cat_zone.info(),
        );

        let actual_tsig_key_name = get_key_for_req(req);
        let expected_tsig_key_name = tsig_key.map(|(name, _alg)| name);
        let tsig_key_mismatch = match (expected_tsig_key_name, actual_tsig_key_name) {
            (None, Some(actual)) => {
                Some(format!(
                    "Request was signed with TSIG key '{actual}' but should be unsigned."))
            }
            (Some(expected), None) => {
                Some(
                    format!("Request should be signed with TSIG key '{expected}' but was unsigned"))
            }
            (Some(expected), Some(actual)) if actual != expected => {
                Some(format!(
                    "Request should be signed with TSIG key '{expected}' but was instead signed with TSIG key '{actual}'"))
            }
            (Some(expected), Some(_)) => {
                trace!("Request is signed with expected TSIG key '{expected}'");
                None
            },
            (None, None) => {
                trace!("Request is unsigned as expected");
                None
            }
        };
        if let Some(reason) = tsig_key_mismatch {
            warn!(
                "{} for {qname} from {} refused: {}",
                q.qtype(),
                req.client_addr(),
                reason
            );
            let response =
                mk_error_response(req.message(), OptRcode::NOTAUTH);
            return ControlFlow::Break(Self::to_stream(response));
        }

        // Read the zone SOA RR
        let read = zone.read();
        let Ok(zone_soa_answer) = Self::read_soa(&read, qname.clone()).await
        else {
            warn!(
                "{} for {qname} from {} refused: zone lacks SOA RR",
                q.qtype(),
                req.client_addr()
            );
            return ControlFlow::Break(Self::to_stream(mk_error_response(
                msg,
                OptRcode::SERVFAIL,
            )));
        };

        match q.qtype() {
            Rtype::SOA => {
                let builder = mk_builder_for_target();
                let response = zone_soa_answer.to_message(msg, builder);
                ControlFlow::Break(Self::to_stream(response))
            }

            Rtype::AXFR | Rtype::IXFR if xfr_settings.is_none() => {
                ControlFlow::Break(Self::to_stream(mk_error_response(
                    msg,
                    OptRcode::REFUSED,
                )))
            }

            Rtype::AXFR => {
                // https://datatracker.ietf.org/doc/html/rfc5936#section-4.2
                // 4.2.  UDP
                //   "With the addition of EDNS0 and applications that require many
                //    small zones, such as in web hosting and some ENUM scenarios,
                //    AXFR sessions on UDP would now seem desirable.  However, there
                //    are still some aspects of AXFR sessions that are not easily
                //    translated to UDP.
                //
                //    Therefore, this document does not update RFC 1035 in this
                //    respect: AXFR sessions over UDP transport are not defined."
                if req.transport_ctx().is_udp() {
                    info!(
                        "AXFR for {qname} from {} refused: not supported over UDP",
                        req.client_addr()
                    );
                    ControlFlow::Break(Self::to_stream(mk_error_response(
                        msg,
                        OptRcode::NOTIMP,
                    )))
                } else {
                    info!("AXFR for {qname} from {}", req.client_addr());
                    let stream = Self::do_axfr(
                        zone_walking_semaphore,
                        batcher_semaphore,
                        req,
                        qname,
                        &zone_soa_answer,
                        cat_zone.info(),
                        read,
                    )
                    .await
                    .unwrap_or_else(|rcode| {
                        Self::to_stream(mk_error_response(msg, rcode))
                    });
                    ControlFlow::Break(stream)
                }
            }

            Rtype::IXFR => {
                info!("IXFR for {qname} from {}", req.client_addr());

                // https://datatracker.ietf.org/doc/html/rfc1995#section-2
                // 2. Brief Description of the Protocol
                //   "Transport of a query may be by either UDP or TCP.  If an
                //    IXFR query is via UDP, the IXFR server may attempt to reply
                //    using UDP if the entire response can be contained in a
                //    single DNS packet.  If the UDP reply does not fit, the query
                //    is responded to with a single SOA record of the server's
                //    current version to inform the client that a TCP query should
                //    be initiated."

                match Self::do_ixfr(
                    batcher_semaphore.clone(),
                    req,
                    qname.clone(),
                    &zone_soa_answer,
                    cat_zone.info(),
                    xfr_mode,
                )
                .await
                {
                    IxfrResult::Ok(stream) => ControlFlow::Break(stream),
                    IxfrResult::FallbackToAxfr => {
                        if xfr_settings.unwrap().strategy
                            != XfrStrategy::IxfrWithAxfrFallback
                        {
                            info!(
                                "IXFR for {qname} from {} refused: client is not permitted to fallback to AXFR",
                                req.client_addr()
                            );
                            ControlFlow::Break(Self::to_stream(
                                mk_error_response(msg, OptRcode::REFUSED),
                            ))
                        } else {
                            info!(
                                "IXFR for {qname} from {}: falling back to AXFR",
                                req.client_addr()
                            );

                            // https://datatracker.ietf.org/doc/html/rfc1995#section-4
                            // 4. Response Format
                            //    "If incremental zone transfer is not available, the
                            //     entire zone is returned.  The first and the last RR
                            //     of the response is the SOA record of the zone.
                            //     I.e. the behavior is the same as an AXFR response
                            //     except the query type is IXFR."
                            let stream = Self::do_axfr(
                                zone_walking_semaphore,
                                batcher_semaphore,
                                req,
                                qname,
                                &zone_soa_answer,
                                cat_zone.info(),
                                read,
                            )
                            .await
                            .unwrap_or_else(|rcode| {
                                Self::to_stream(mk_error_response(msg, rcode))
                            });
                            ControlFlow::Break(stream)
                        }
                    }

                    IxfrResult::Err(rcode) => ControlFlow::Break(
                        Self::to_stream(mk_error_response(msg, rcode)),
                    ),
                }
            }

            _ => ControlFlow::Continue(()),
        }
    }

    fn settings_for_client<'a>(
        client_ip: IpAddr,
        qname: &Name<Bytes>,
        qtype: Rtype,
        zone_info: &'a ZoneInfo,
    ) -> (Option<&'a XfrSettings>, Option<&'a TsigKey>) {
        let ZoneType::Primary { allow_xfr, .. } = zone_info.zone_type()
        else {
            if qtype != Rtype::SOA {
                warn!(
                    "{qtype} for {qname} from {client_ip} refused: zone does not allow XFR",
                );
            }
            return (None, None);
        };

        let Some((xfr_settings, tsig_key)) = allow_xfr.get_caller(client_ip)
        else {
            if qtype != Rtype::SOA {
                warn!(
                    "{qtype} for {qname} from {client_ip} refused: client is not permitted to transfer this zone",
                );
            }
            return (None, None);
        };

        if matches!(
            (qtype, xfr_settings.strategy),
            (Rtype::AXFR, XfrStrategy::IxfrOnly)
                | (Rtype::IXFR, XfrStrategy::AxfrOnly)
        ) {
            if qtype != Rtype::SOA {
                warn!(
                    "{qtype} for {qname} from {client_ip} refused: zone does not allow {qtype}",
                );
            }
            return (None, None);
        }

        (Some(xfr_settings), tsig_key.as_ref())
    }

    #[allow(clippy::too_many_arguments)]
    async fn do_axfr<T>(
        zone_walk_semaphore: Arc<Semaphore>,
        batcher_semaphore: Arc<Semaphore>,
        req: &Request<RequestOctets, T>,
        qname: Name<Bytes>,
        zone_soa_answer: &Answer,
        zone_info: &ZoneInfo,
        read: Box<dyn ReadableZone>,
    ) -> Result<
        XfrMiddlewareStream<
            NextSvc::Future,
            NextSvc::Stream,
            <NextSvc::Stream as Stream>::Item,
        >,
        OptRcode,
    > {
        let msg = req.message();

        let AnswerContent::Data(zone_soa_rrset) =
            zone_soa_answer.content().clone()
        else {
            unreachable!()
        };

        let ZoneType::Primary { allow_xfr, .. } = zone_info.zone_type()
        else {
            unreachable!();
        };

        let Some((xfr_settings, _tsig_key)) =
            allow_xfr.get_caller(req.client_addr().ip())
        else {
            unreachable!();
        };

        let compatibility_mode = xfr_settings.compatibility_mode
            == CompatibilityMode::BackwardCompatible;

        if compatibility_mode {
            trace!(
                "Compatibility mode enabled for client with IP address {}",
                req.client_addr().ip()
            );
        }

        // Return a stream of response messages containing:
        //   - SOA
        //   - RRSETs, one or more per response message
        //   - SOA
        //
        // Neither RFC 5936 nor RFC 1035 defined AXFR for UDP, only for TCP.
        // However, RFC 1995 says that for IXFR if no diffs are available the
        // full zone should be served just as with AXFR, and that UDP is
        // supported as long as the entire XFR response fits in a single
        // datagram. Thus we don't check for UDP or TCP here, except to abort
        // if the response is too large to fit in a single UDP datagram,
        // instead we let the caller that has the context decide whether AXFR
        // is supported or not.
        //
        // References:
        //   - https://datatracker.ietf.org/doc/html/rfc1995#section-2
        //   - https://datatracker.ietf.org/doc/html/rfc1995#section-4
        //   - https://datatracker.ietf.org/doc/html/rfc5936#section-4.2

        // Create a stream that will be immediately returned to the caller.
        // Async tasks will then push DNS response messages into the stream as
        // they become available.
        let (sender, receiver) = unbounded_channel();
        let stream = UnboundedReceiverStream::new(receiver);

        // Create a bounded queue for passing RRsets found during zone walking
        // to a task which will batch the RRs together before pushing them
        // into the result stream.
        let (batcher_tx, mut batcher_rx) =
            tokio::sync::mpsc::channel::<(StoredName, SharedRrset)>(100);

        // Notify the underlying transport to expect a stream of related
        // responses. The transport should modify its behaviour to account for
        // the potentially slow and long running nature of a transaction.
        Self::add_to_stream(
            CallResult::feedback_only(ServiceFeedback::BeginTransaction),
            &sender,
        );

        // Enqueue the zone SOA RRset for the batcher to process.
        if batcher_tx
            .send((qname.clone(), zone_soa_rrset.clone()))
            .await
            .is_err()
        {
            return Err(OptRcode::SERVFAIL);
        }

        // Stream the remaining non-SOA zone RRsets in the background to the
        // batcher.
        tokio::spawn(async move {
            // Limit the number of concurrently running XFR related zone
            // walking operations.
            if zone_walk_semaphore.acquire().await.is_err() {
                error!("Internal error: Failed to acquire XFR zone walking semaphore");
                return;
            }

            let cloned_batcher_tx = batcher_tx.clone();
            let op =
                Box::new(move |owner: StoredName, rrset: &SharedRrset| {
                    if rrset.rtype() != Rtype::SOA {
                        let _ = cloned_batcher_tx
                            .blocking_send((owner.clone(), rrset.clone()));
                        // If the blocking send fails it means that he batcher
                        // is no longer available. This can happen if it was
                        // no longer able to pass messages back to the
                        // underlying transport, which can happen if the
                        // client closed the connection. We don't log this
                        // because we can't stop the tree walk and so will
                        // keep hitting this error until the tree walk is
                        // complete, causing a lot of noise if we were to log
                        // this.
                    }
                });

            // Walk the zone tree, invoking our operation for each leaf.
            match read.is_async() {
                true => {
                    read.walk_async(op).await;
                    if let Err(err) =
                        batcher_tx.send((qname, zone_soa_rrset)).await
                    {
                        error!("Internal error: Failed to send final AXFR SOA to batcher: {err}");
                    }
                }
                false => {
                    tokio::task::spawn_blocking(move || {
                        read.walk(op);
                        if let Err(err) =
                            batcher_tx.blocking_send((qname, zone_soa_rrset))
                        {
                            error!("Internal error: Failed to send final AXFR SOA to batcher: {err}");
                        }
                    });
                }
            }
        });

        // Combine RRsets enumerated by zone walking as many as possible per
        // DNS response message and pass the created messages downstream to
        // the caller.
        let msg = msg.clone();

        let soft_byte_limit = match req.transport_ctx() {
            TransportSpecificContext::Udp(ctx) => {
                let max_msg_size =
                    ctx.max_response_size_hint().unwrap_or(512);
                max_msg_size - req.num_reserved_bytes()
            }
            TransportSpecificContext::NonUdp(_) => {
                65535 - req.num_reserved_bytes()
            }
        };

        tokio::spawn(async move {
            // Limit the number of concurrently running XFR batching
            // operations.
            if batcher_semaphore.acquire().await.is_err() {
                error!(
                    "Internal error: Failed to acquire XFR batcher semaphore"
                );
                return;
            }

            let Ok(qclass) = msg.sole_question().map(|q| q.qclass()) else {
                unreachable!();
            };

            let (mut owner, mut zone_soa_rrset) =
                batcher_rx.recv().await.unwrap();
            assert_eq!(zone_soa_rrset.rtype(), Rtype::SOA);
            let saved_soa_rrset = zone_soa_rrset.clone();
            let mut soa_seen_count = 0;

            let mut current_rrset = zone_soa_rrset;
            let mut records_to_process = current_rrset.data();

            // Note: NSD apparently uses name compresson on AXFR responses
            // because AXFR responses they typically contain lots of
            // alphabetically ordered duplicate names which compress well. NSD
            // limits AXFR responses to 16,383 RRs because DNS name
            // compression uses a 14-bit offset (2^14-1=16383) from the start
            // of the message to the first occurence of a name instead of
            // repeating the name, and name compression is less effective
            // over 16383 bytes. (Credit: Wouter Wijngaards)
            //
            // TODO: Once we start supporting name compression in responses decide
            // if we want to behave the same way.

            let mut batcher = RrBatcher::new(msg.clone());
            if compatibility_mode {
                batcher.set_hard_rr_limit(1);
            }

            batcher.set_soft_byte_limit(soft_byte_limit);

            // Loop until all of the RRsets sent by the zone walker have been
            // pushed into DNS response messages and sent into the result
            // stream.
            'outer: loop {
                // Loop over the RRset items being sent by the zone walker and
                // add as many of them as possible to the response being
                // built.

                let mut num_rrs_added = 0;
                let builder = 'inner: loop {
                    if records_to_process == saved_soa_rrset.data() {
                        soa_seen_count += 1;
                    }

                    for rr in records_to_process {
                        let res = batcher.push((
                            owner.clone(),
                            qclass,
                            current_rrset.ttl(),
                            rr,
                        ));

                        match res {
                            Ok(PushResult::PushedAndReadyForMore) => {
                                // Message still has space, keep going.
                                num_rrs_added += 1;
                            }

                            Ok(PushResult::PushedAndLimitReached(
                                builder,
                            )) => {
                                // Pushed and configured limit reached, send it.
                                num_rrs_added += 1;
                                trace!("Pushed and limit reached");
                                break 'inner Some(builder);
                            }

                            Ok(PushResult::NotPushedMessageFull(builder)) => {
                                // Message is full, send what we have so far.
                                trace!("Message is full");
                                break 'inner Some(builder);
                            }

                            Err(err) => {
                                error!("Internal error: Unable to add RR to AXFR response message: {err}");
                                break 'outer;
                            }
                        }
                    }

                    if soa_seen_count == 2 {
                        trace!("SOA seen count == 2, stopping");
                        // This is our signal to stop, as the AXFR message
                        // starts and ends with the zone SOA.
                        break 'inner batcher.take();
                    } else {
                        // Fetch more RRsets to add to the message.
                        (owner, current_rrset) =
                            batcher_rx.recv().await.unwrap();
                        records_to_process = current_rrset.data();
                    }
                };

                // Send the message.
                if let Some(builder) = builder {
                    trace!("Sending");
                    let mut additional = builder.additional();
                    Self::set_axfr_header(&msg, &mut additional);
                    let call_result = Ok(CallResult::new(additional));
                    if sender.send(call_result).is_err() {
                        batcher_rx.close();
                        return;
                    }
                }

                if (num_rrs_added as usize) < records_to_process.len() {
                    // Some RRs from the current RRset didn't fit in the last
                    // message, process these before fetching another RRset
                    // from the incoming queue.
                    records_to_process =
                        &current_rrset.data()[(num_rrs_added as usize)..];
                } else if soa_seen_count == 2 {
                    trace!("SOA seen count == 2, really stopping");
                    break;
                } else {
                    // Fetch more RRsets to add to the message from the incoming queue.
                    trace!("Fetching more records");
                    let res = batcher_rx.recv().await;
                    if res.is_none() {
                        trace!("Fetch failed");
                        let mut additional =
                            mk_error_response(&msg, OptRcode::SERVFAIL);
                        Self::set_axfr_header(&msg, &mut additional);
                        let call_result = Ok(CallResult::new(additional));
                        trace!("Sending");
                        sender.send(call_result).unwrap(); // TODO: Handle this Result
                        break 'outer;
                    };
                    (owner, zone_soa_rrset) = res.unwrap();
                    records_to_process = zone_soa_rrset.data();
                }
            }

            trace!("Finishing transaction");
            Self::add_to_stream(
                CallResult::feedback_only(ServiceFeedback::EndTransaction),
                &sender,
            );

            batcher_rx.close();
        });

        Ok(MiddlewareStream::Result(stream))
    }

    // Returns None if fallback to AXFR should be done.
    #[allow(clippy::too_many_arguments)]
    async fn do_ixfr<T>(
        batcher_semaphore: Arc<Semaphore>,
        req: &Request<RequestOctets, T>,
        qname: Name<Bytes>,
        zone_soa_answer: &Answer,
        zone_info: &ZoneInfo,
        xfr_mode: XfrMode,
    ) -> IxfrResult<
        XfrMiddlewareStream<
            NextSvc::Future,
            NextSvc::Stream,
            <NextSvc::Stream as Stream>::Item,
        >,
    > {
        if xfr_mode == XfrMode::AxfrOnly {
            trace!("Not responding with IXFR as mode is set to AXFR only");
            return IxfrResult::FallbackToAxfr;
        }

        let msg = req.message();

        // https://datatracker.ietf.org/doc/html/rfc1995#section-2
        // 2. Brief Description of the Protocol
        //   "Transport of a query may be by either UDP or TCP.  If an IXFR
        //    query is via UDP, the IXFR server may attempt to reply using UDP
        //    if the entire response can be contained in a single DNS packet.
        //    If the UDP reply does not fit, the query is responded to with a
        //    single SOA record of the server's current version to inform the
        //    client that a TCP query should be initiated."
        //
        // https://datatracker.ietf.org/doc/html/rfc1995#section-3
        // 3. Query Format
        //   "The IXFR query packet format is the same as that of a normal DNS
        //    query, but with the query type being IXFR and the authority
        //    section containing the SOA record of client's version of the
        //    zone."
        //
        // https://datatracker.ietf.org/doc/html/rfc1995#section-4
        // 4. Response Format
        //   "If incremental zone transfer is not available, the entire zone
        //    is returned.  The first and the last RR of the response is the
        //    SOA record of the zone.  I.e. the behavior is the same as an
        //    AXFR response except the query type is IXFR."
        //
        // https://datatracker.ietf.org/doc/html/rfc1995#section-2
        // 2. Brief Description of the Protocol
        //   "To ensure integrity, servers should use UDP checksums for all
        //    UDP responses.  A cautious client which receives a UDP packet
        //    with a checksum value of zero should ignore the result and try a
        //    TCP IXFR instead."
        if let Ok(mut query_soas) = msg
            .authority()
            .map(|section| section.limit_to::<Soa<ParsedName<_>>>())
        {
            if let Some(Ok(query_soa)) = query_soas.next() {
                if query_soas.next().is_none() {
                    let query_serial = query_soa.data().serial();

                    if let AnswerContent::Data(rrset) =
                        zone_soa_answer.content()
                    {
                        if rrset.data().len() == 1 {
                            if let ZoneRecordData::Soa(soa) =
                                rrset.first().unwrap().data()
                            {
                                let zone_serial = soa.serial();

                                // TODO: if cached then return cached IXFR response
                                return Self::compute_ixfr(
                                    batcher_semaphore,
                                    req,
                                    qname,
                                    query_serial,
                                    zone_serial,
                                    zone_info,
                                    zone_soa_answer,
                                )
                                .await;
                            }
                        }
                    }

                    return IxfrResult::Err(OptRcode::SERVFAIL);
                }
            }
        }

        IxfrResult::Err(OptRcode::FORMERR)
    }

    #[allow(clippy::too_many_arguments)]
    async fn compute_ixfr<T>(
        batcher_semaphore: Arc<Semaphore>,
        req: &Request<RequestOctets, T>,
        qname: Name<Bytes>,
        query_serial: Serial,
        zone_serial: Serial,
        zone_info: &ZoneInfo,
        zone_soa_answer: &Answer,
    ) -> IxfrResult<
        XfrMiddlewareStream<
            NextSvc::Future,
            NextSvc::Stream,
            <NextSvc::Stream as Stream>::Item,
        >,
    > {
        let msg = req.message();

        let AnswerContent::Data(zone_soa_rrset) =
            zone_soa_answer.content().clone()
        else {
            unreachable!()
        };

        // Note: Unlike RFC 5936 for AXFR, neither RFC 1995 nor RFC 9103 say
        // anything about whether an IXFR response can consist of more than
        // one response message, but given the 2^16 byte maximum response size
        // of a TCP DNS message and the 2^16 maximum number of ANSWER RRs
        // allowed per DNS response, large zones may not fit in a single
        // response message and will have to be split into multiple response
        // messages.

        // https://datatracker.ietf.org/doc/html/rfc1995#section-2
        // 2. Brief Description of the Protocol
        //   "If an IXFR query with the same or newer version number than that
        //    of the server is received, it is replied to with a single SOA
        //    record of the server's current version, just as in AXFR."
        //                                            ^^^^^^^^^^^^^^^
        // Errata https://www.rfc-editor.org/errata/eid3196 points out that
        // this is NOT "just as in AXFR" as AXFR does not do that.
        if query_serial >= zone_serial {
            let builder = mk_builder_for_target();
            let response = zone_soa_answer.to_message(msg, builder);
            trace!("IXFR finished because query_serial >= zone_serial");
            return IxfrResult::Ok(Self::to_stream(response));
        }

        // Get the necessary diffs, if available
        let start_serial = query_serial;
        let end_serial = zone_serial;
        let diffs = zone_info.diffs_for_range(start_serial, end_serial).await;
        if diffs.is_empty() {
            return IxfrResult::FallbackToAxfr;
        };

        // TODO: Add something like the Bind `max-ixfr-ratio` option that
        // "sets the size threshold (expressed as a percentage of the size of
        // the full zone) beyond which named chooses to use an AXFR response
        // rather than IXFR when answering zone transfer requests"?

        // Create a stream that will be immediately returned to the caller.
        // Async tasks will then push DNS response messages into the stream as
        // they become available.
        let (sender, receiver) = unbounded_channel();
        let stream = UnboundedReceiverStream::new(receiver);

        // Notify the underlying transport to expect a stream of related
        // responses. The transport should modify its behaviour to account for
        // the potentially slow and long running nature of a transaction.
        Self::add_to_stream(
            CallResult::feedback_only(ServiceFeedback::BeginTransaction),
            &sender,
        );

        // Stream the IXFR diffs in the background
        let msg = msg.clone();
        tokio::spawn(async move {
            // Limit the number of concurrently running XFR batching
            // operations.
            if batcher_semaphore.acquire().await.is_err() {
                error!(
                    "Internal error: Failed to acquire XFR batcher semaphore"
                );
                return;
            }

            // https://datatracker.ietf.org/doc/html/rfc1995#section-4
            // 4. Response Format
            //    ...
            //   "If incremental zone transfer is available, one or more
            //    difference sequences is returned.  The list of difference
            //    sequences is preceded and followed by a copy of the server's
            //    current version of the SOA.
            //
            //    Each difference sequence represents one update to the zone
            //    (one SOA serial change) consisting of deleted RRs and added
            //    RRs.  The first RR of the deleted RRs is the older SOA RR
            //    and the first RR of the added RRs is the newer SOA RR.
            //
            //    Modification of an RR is performed first by removing the
            //    original RR and then adding the modified one.
            //
            //    The sequences of differential information are ordered oldest
            //    first newest last.  Thus, the differential sequences are the
            //    history of changes made since the version known by the IXFR
            //    client up to the server's current version.
            //
            //    RRs in the incremental transfer messages may be partial. That
            //    is, if a single RR of multiple RRs of the same RR type changes,
            //    only the changed RR is transferred."

            let (owner, qclass) = {
                let Ok(q) = msg.sole_question() else {
                    unreachable!();
                };
                (q.qname().to_name::<Bytes>(), q.qclass())
            };

            let diff_sequences = diffs
                .iter()
                .flat_map(|diff| {
                    // https://datatracker.ietf.org/doc/html/rfc1995#section-4
                    // 4. Response Format
                    //    "Each difference sequence represents one update to the zone
                    //    (one SOA serial change) consisting of deleted RRs and added
                    //    RRs.  The first RR of the deleted RRs is the older SOA RR
                    //    and the first RR of the added RRs is the newer SOA RR.
                    let deleted = diff.removed.get(&qname).unwrap();
                    let added = diff.added.get(&qname).unwrap();

                    let mut rrset_iters: std::vec::Vec<
                        Box<dyn Iterator<Item = SharedRrset>>,
                    > = vec![];

                    for rrsets in [deleted, added] {
                        rrset_iters.push(Box::new(std::iter::once(
                            rrsets
                                .iter()
                                .find(|rrset| rrset.rtype() == Rtype::SOA)
                                .unwrap() // The zone MUST have a SOA record
                                .clone(),
                        )));

                        rrset_iters.push(Box::new(
                            rrsets
                                .iter()
                                .filter(|rrset| rrset.rtype() != Rtype::SOA)
                                .cloned(),
                        ));
                    }

                    rrset_iters
                })
                .flatten();

            let rrsets = std::iter::once(zone_soa_rrset.clone())
                .chain(diff_sequences)
                .chain(std::iter::once(zone_soa_rrset.clone()));

            let mut batcher = RrBatcher::new(msg.clone());

            // Loop until all of the RRsets in the diffs have been pushed into
            // DNS response messages and sent into the result stream.
            let mut records_to_process;
            let mut num_rrs_added;

            for rrset in rrsets {
                trace!("Starting RRSET.. ({} records)", rrset.data().len());
                records_to_process = rrset.data();
                num_rrs_added = 0;

                'outer: loop {
                    let builder = 'inner: loop {
                        for rr in records_to_process {
                            match batcher.push((
                                owner.clone(),
                                qclass,
                                rrset.ttl(),
                                rr,
                            )) {
                                Ok(PushResult::PushedAndReadyForMore) => {
                                    // Message still has space, keep going.
                                    num_rrs_added += 1;
                                }

                                Ok(PushResult::PushedAndLimitReached(
                                    builder,
                                )) => {
                                    // Pushed and configured limit reached, send it.
                                    num_rrs_added += 1;
                                    trace!("Pushed and limit reached");
                                    break 'inner builder;
                                }

                                Ok(PushResult::NotPushedMessageFull(
                                    builder,
                                )) => {
                                    // Message is full, send what we have so far.
                                    trace!("Message is full");
                                    break 'inner builder;
                                }

                                Err(err) => {
                                    error!("Internal error: Unable to add RR to IXFR response message: {err}");
                                    let mut additional = mk_error_response(
                                        &msg,
                                        OptRcode::SERVFAIL,
                                    );
                                    Self::set_axfr_header(
                                        &msg,
                                        &mut additional,
                                    );
                                    let call_result =
                                        Ok(CallResult::new(additional));
                                    let _ = sender.send(call_result); // TODO: Handle this Result.
                                    break 'outer;
                                }
                            }
                        }

                        if rrset.data().len() > num_rrs_added {
                            trace!(
                                "More to process... ({num_rrs_added} / {})",
                                rrset.data().len()
                            );
                            records_to_process =
                                &rrset.data()[num_rrs_added..];
                        } else {
                            trace!("end of RRSET reached");
                            break 'outer;
                        }
                    };

                    trace!("sending intermediate batch");
                    let mut additional = builder.additional();
                    Self::set_axfr_header(&msg, &mut additional);
                    let call_result = Ok(CallResult::new(additional));
                    let _ = sender.send(call_result); // TODO: Handle this Result.
                }
            }

            if let Some(builder) = batcher.take() {
                trace!("sending final batch");
                let mut additional = builder.additional();
                Self::set_axfr_header(&msg, &mut additional);
                let call_result = Ok(CallResult::new(additional));
                let _ = sender.send(call_result); // TODO: Handle this Result.
            }

            trace!("Ending transaction");
            Self::add_to_stream(
                CallResult::feedback_only(ServiceFeedback::EndTransaction),
                &sender,
            );
        });

        IxfrResult::Ok(MiddlewareStream::Result(stream))
    }

    fn add_to_stream(
        call_result: CallResult<NextSvc::Target>,
        sender: &UnboundedSender<ServiceResult<NextSvc::Target>>,
    ) {
        sender.send(Ok(call_result)).unwrap(); // TODO: Handle this Result
    }

    fn set_axfr_header(
        msg: &Message<RequestOctets>,
        additional: &mut AdditionalBuilder<StreamTarget<NextSvc::Target>>,
    ) {
        // https://datatracker.ietf.org/doc/html/rfc5936#section-2.2.1
        // 2.2.1: Header Values
        //
        // "These are the DNS message header values for AXFR responses.
        //
        //     ID          MUST be copied from request -- see Note a)
        //
        //     QR          MUST be 1 (Response)
        //
        //     OPCODE      MUST be 0 (Standard Query)
        //
        //     Flags:
        //        AA       normally 1 -- see Note b)
        //        TC       MUST be 0 (Not truncated)
        //        RD       RECOMMENDED: copy request's value; MAY be set to 0
        //        RA       SHOULD be 0 -- see Note c)
        //        Z        "mbz" -- see Note d)
        //        AD       "mbz" -- see Note d)
        //        CD       "mbz" -- see Note d)"
        let header = additional.header_mut();

        // Note: MandatoryMiddlewareSvc will also "fix" ID and QR, so strictly
        // speaking this isn't necessary, but as a caller might not use
        // MandatoryMiddlewareSvc we do it anyway to try harder to conform to
        // the RFC.
        header.set_id(msg.header().id());
        header.set_qr(true);

        header.set_opcode(Opcode::QUERY);
        header.set_aa(true);
        header.set_tc(false);
        header.set_rd(msg.header().rd());
        header.set_ra(false);
        header.set_z(false);
        header.set_ad(false);
        header.set_cd(false);
    }

    fn to_stream(
        response: AdditionalBuilder<StreamTarget<NextSvc::Target>>,
    ) -> XfrMiddlewareStream<
        NextSvc::Future,
        NextSvc::Stream,
        <NextSvc::Stream as Stream>::Item,
    > {
        let res = Ok(CallResult::new(response));
        MiddlewareStream::Map(once(ready(res)))
    }

    #[allow(clippy::borrowed_box)]
    async fn read_soa(
        read: &Box<dyn ReadableZone>,
        qname: Name<Bytes>,
    ) -> Result<Answer, OutOfZone> {
        match read.is_async() {
            true => read.query_async(qname, Rtype::SOA).await,
            false => read.query(qname, Rtype::SOA),
        }
    }

    fn get_relevant_question(
        msg: &Message<RequestOctets>,
    ) -> Option<Question<ParsedName<RequestOctets::Range<'_>>>> {
        if Opcode::QUERY == msg.header().opcode() && !msg.header().qr() {
            if let Ok(q) = msg.sole_question() {
                if matches!(q.qtype(), Rtype::SOA | Rtype::AXFR | Rtype::IXFR)
                {
                    return Some(q);
                }
            }
        }

        None
    }
}

//--- Service (with TSIG key name in the request metadata)

impl<RequestOctets, NextSvc, KS> Service<RequestOctets, Option<KeyName>>
    for XfrMiddlewareSvc<RequestOctets, NextSvc, KS>
where
    RequestOctets: Octets + Send + Sync + Unpin + 'static,
    for<'a> <RequestOctets as octseq::Octets>::Range<'a>: Send + Sync,
    NextSvc: Service<RequestOctets, ()> + Clone + Send + Sync + 'static,
    NextSvc::Future: Send + Sync + Unpin,
    NextSvc::Target: Composer + Default + Send + Sync,
    NextSvc::Stream: Send + Sync,
    KS: KeyStore + Clone + Send + Sync + 'static,
    KS::Key: Send + Sync,
{
    type Target = NextSvc::Target;
    type Stream = XfrMiddlewareStream<
        NextSvc::Future,
        NextSvc::Stream,
        <NextSvc::Stream as Stream>::Item,
    >;
    type Future = Pin<Box<dyn Future<Output = Self::Stream> + Send + Sync>>;

    fn call(
        &self,
        request: Request<RequestOctets, Option<KeyName>>,
    ) -> Self::Future {
        let request = request.clone();
        let next_svc = self.next_svc.clone();
        let catalog = self.catalog.clone();
        let zone_walking_semaphore = self.zone_walking_semaphore.clone();
        let batcher_semaphore = self.batcher_semaphore.clone();
        let xfr_mode = self.xfr_mode;
        Box::pin(async move {
            match Self::preprocess(
                zone_walking_semaphore,
                batcher_semaphore,
                &request,
                catalog,
                xfr_mode,
                |req| req.metadata().as_ref(),
            )
            .await
            {
                ControlFlow::Continue(()) => {
                    let request = request.with_new_metadata(());
                    let stream = next_svc.call(request).await;
                    MiddlewareStream::IdentityStream(stream)
                }
                ControlFlow::Break(stream) => stream,
            }
        })
    }
}

//--- Service (without TSIG key name in the request metadata)

impl<RequestOctets, NextSvc, KS> Service<RequestOctets, ()>
    for XfrMiddlewareSvc<RequestOctets, NextSvc, KS>
where
    RequestOctets: Octets + Send + Sync + Unpin + 'static,
    for<'a> <RequestOctets as octseq::Octets>::Range<'a>: Send + Sync,
    NextSvc: Service<RequestOctets, ()> + Clone + Send + Sync + 'static,
    NextSvc::Future: Send + Sync + Unpin,
    NextSvc::Target: Composer + Default + Send + Sync,
    NextSvc::Stream: Send + Sync,
    KS: KeyStore + Clone + Send + Sync + 'static,
    KS::Key: Send + Sync,
{
    type Target = NextSvc::Target;
    type Stream = XfrMiddlewareStream<
        NextSvc::Future,
        NextSvc::Stream,
        <NextSvc::Stream as Stream>::Item,
    >;
    type Future = Pin<Box<dyn Future<Output = Self::Stream> + Send + Sync>>;

    fn call(&self, request: Request<RequestOctets, ()>) -> Self::Future {
        let request = request.clone();
        let next_svc = self.next_svc.clone();
        let catalog = self.catalog.clone();
        let zone_walking_semaphore = self.zone_walking_semaphore.clone();
        let batcher_semaphore = self.batcher_semaphore.clone();
        let xfr_mode = self.xfr_mode;
        Box::pin(async move {
            match Self::preprocess(
                zone_walking_semaphore,
                batcher_semaphore,
                &request,
                catalog,
                xfr_mode,
                |_| None,
            )
            .await
            {
                ControlFlow::Continue(()) => {
                    let request = request.with_new_metadata(());
                    let stream = next_svc.call(request).await;
                    MiddlewareStream::IdentityStream(stream)
                }
                ControlFlow::Break(stream) => stream,
            }
        })
    }
}

//----------- PushResult ------------------------------------------------------

enum PushResult<Target> {
    PushedAndReadyForMore,
    PushedAndLimitReached(AnswerBuilder<StreamTarget<Target>>),
    NotPushedMessageFull(AnswerBuilder<StreamTarget<Target>>),
}

//----------- RrBatcher -------------------------------------------------------

// IDEA: Maybe this should act like an iterator and whenever it runs out of
// items to process one should feed it more? And pass it RRsets instead of
// Records.

struct RrBatcher<RequestOctets, Target> {
    req_msg: Arc<Message<RequestOctets>>,
    hard_rr_limit: Option<u16>,
    soft_byte_limit: Option<usize>,
    answer: Option<Result<AnswerBuilder<StreamTarget<Target>>, PushError>>,
}

impl<RequestOctets, Target> RrBatcher<RequestOctets, Target>
where
    RequestOctets: Octets,
    Target: Composer + Default,
{
    pub fn new(req_msg: Arc<Message<RequestOctets>>) -> Self {
        Self {
            req_msg,
            hard_rr_limit: None,
            soft_byte_limit: None,
            answer: None,
        }
    }

    pub fn set_hard_rr_limit(&mut self, rr_limit: u16) {
        self.hard_rr_limit = Some(rr_limit);
    }

    pub fn set_soft_byte_limit(&mut self, byte_limit: u16) {
        trace!("Setting soft byte limit to {byte_limit} bytes");
        self.soft_byte_limit = Some(byte_limit as usize);
    }

    pub fn push(
        &mut self,
        record: impl ComposeRecord,
    ) -> Result<PushResult<Target>, PushError> {
        self.answer.get_or_insert_with(|| {
            let mut builder = mk_builder_for_target();
            if let Some(limit) = self.soft_byte_limit {
                builder.set_push_limit(limit);
            }
            builder.start_answer(&self.req_msg, Rcode::NOERROR)
        });

        let mut answer = self.answer.take().unwrap()?;

        let res = answer.push(record);
        let ancount = answer.counts().ancount();
        // let msg_len = answer.as_slice().len() as u16;

        match res {
            Ok(()) if Some(ancount) == self.hard_rr_limit => {
                // Push succeeded but the message is as full as the caller
                // allows, pass it back to the caller to process.
                Ok(PushResult::PushedAndLimitReached(answer))
            }

            // Ok(())
            //     if self.soft_byte_limit.is_some()
            //         && Some(msg_len) >= self.soft_byte_limit =>
            // {
            //     // Push succeeded but the message is as full as the caller
            //     // allows, pass it back to the caller to process.
            //     Ok(PushResult::PushedAndLimitReached(answer))
            // }
            Err(_) if ancount > 0 => {
                // Push failed because the message is full, pass it back to
                // the caller to process.
                Ok(PushResult::NotPushedMessageFull(answer))
            }

            Err(err) => {
                // We expect to be able to add at least one answer to the message.
                Err(err)
            }

            Ok(()) => {
                // Record has been added, keep the answer builder for the next push.
                self.answer = Some(Ok(answer));
                Ok(PushResult::PushedAndReadyForMore)
            }
        }
    }

    pub fn take(&mut self) -> Option<AnswerBuilder<StreamTarget<Target>>> {
        self.answer.take().and_then(|res| res.ok())
    }
}

//------------ IxfrResult -----------------------------------------------------

enum IxfrResult<Stream> {
    Ok(Stream),
    FallbackToAxfr,
    Err(OptRcode),
}
