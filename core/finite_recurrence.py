"""Finite observed-path reset-budget diagnostics, not an infinite LBSM proof."""
from __future__ import annotations
import numpy as np

FAILURES = {'drift_pass':'DRIFT_FAIL', 'delta_margin_pass':'DELTA_MARGIN_FAIL',
    'region_gate_pass':'REGION_GATE_FAIL', 'transition_soundness_pass':'TRANSITION_SOUNDNESS_FAIL',
    'event_source_pass':'EVENT_SOURCE_FAIL', 'count_pass':'INSUFFICIENT_EVENTS',
    'CV_pass':'CV_FAIL', 'sliding_pass':'SLIDING_WINDOW_FAIL'}


def safe_window_monitor(safe, length=48):
    """safe[0] is initialization only; events consume disjoint states 1..H.

    An unsafe state interrupts a pending positive run, but DOES NOT reset the
    certificate budget. Only a completed safe window permits a budget reset.
    """
    safe=np.asarray(safe,dtype=bool)
    if safe.ndim!=1 or len(safe)<2 or length<1: raise ValueError('Invalid monitor input')
    counter=np.zeros(len(safe),dtype=np.int64); events=[]; interruptions=0; episodes=0
    run=0
    for t in range(1,len(safe)):
        if safe[t]:
            run+=1
            if run==length: events.append(t); run=0
        else:
            interruptions+=int(run>0); run=0
            episodes+=int(t==1 or safe[t-1])
        counter[t]=run
    return dict(event_times=events,event_count=len(events),counter=counter,
                hazard_interruption_count=interruptions,hazard_episode_count=episodes,
                hazard_state_count=int((~safe[1:]).sum()))


def reference_events(safe,length=48):
    """Independent interval-based detector for timestamp/reset reconciliation."""
    safe=np.asarray(safe,dtype=bool); events=[]; start=1
    for stop in range(1,len(safe)+1):
        if stop==len(safe) or not safe[stop]:
            events.extend(range(start+length-1,stop,length)); start=stop+1
    return events


def segments(events,horizon):
    events=list(map(int,events))
    if events!=sorted(set(events)) or any(t<1 or t>horizon for t in events):
        raise ValueError('Invalid event timeline')
    result=[]; start=0
    for t in events:
        result.append((start,t,True)); start=t
    if start<horizon: result.append((start,horizon,False))
    return result


def path_score(prediction,events,eta=.01):
    """One score per whole path, including the observed right-censored tail.

    Tail score covers only time observed through H; never imputes a next event.
    """
    prediction=np.asarray(prediction,dtype=float); horizon=len(prediction)-1
    return float(max([0.]+[eta*(e-s)-prediction[s] for s,e,_ in segments(events,horizon)]))


def budget_diagnostics(prediction,events,reference,delta,region_mask,eta=.01,tol=1e-7):
    prediction=np.asarray(prediction,dtype=float); horizon=len(prediction)-1
    region_mask=np.asarray(region_mask,dtype=bool)
    if prediction.ndim!=1 or horizon<1 or region_mask.shape!=prediction.shape:
        raise ValueError('State/transition shape mismatch')
    if not np.isfinite(delta) or delta<0 or eta<=0: raise ValueError('Invalid budget parameters')
    segment_list=segments(events,horizon)
    post=np.empty(horizon+1); pre=np.empty(horizon); post[0]=prediction[0]+delta
    event_set=set(events)
    for t in range(horizon):
        pre[t]=post[t]-eta
        post[t+1]=prediction[t+1]+delta if t+1 in event_set else pre[t]
    finite=bool(np.isfinite(post).all() and np.isfinite(pre).all())
    # Check the arrival budget BEFORE any event reset. This includes the last
    # edge into an event and edge H-1 -> H: no masked/artificial terminal edge.
    drift=pre-post[:-1]
    sound=np.isfinite(pre)&np.isfinite(post[:-1])&(pre>=-tol)&(post[:-1]>=-tol)
    reports=[dict(start=s,end=e,completed=complete,length=e-s,
        budget=float(prediction[s]+delta),required_budget=float(eta*(e-s)),
        margin_pass=bool(prediction[s]+delta+tol>=eta*(e-s))) for s,e,complete in segment_list]
    zsource=post[:-1]<eta
    event_source=list(events)==list(reference)
    return dict(drift_pass=bool(finite and (drift<=-eta+tol).all()),
        delta_margin_pass=bool(finite and all(s['margin_pass'] for s in reports)),
        region_gate_pass=bool(region_mask.all() and sound[zsource].all()),
        transition_soundness_pass=bool(finite and sound.all() and (post>=-tol).all() and event_source),
        event_source_pass=bool(event_source),CV_pass=None,sliding_pass=None,
        n_transitions=horizon,n_region_outside_states=int((~region_mask).sum()),
        n_unsound_transitions=int((~sound).sum()),n_zfree_sources=int(zsource.sum()),
        n_zfree_unsound=int((~sound[zsource]).sum()),
        zfree_status='CHECKED' if zsource.any() else 'NO_OBSERVED_SOURCES',
        min_pre_reset_budget=float(pre.min()),segments=reports,
        budget_pre_reset=pre.tolist(),budget_post_reset=post.tolist())


def combine_gates(flags,event_count,m_min,*,cv_enabled=False,sliding_enabled=False):
    if m_min<1: raise ValueError('Positive event count threshold required')
    checks={k:flags.get(k) for k in FAILURES}
    checks['count_pass']=event_count>=m_min
    active=set(FAILURES)-({'CV_pass'} if not cv_enabled else set())-({'sliding_pass'} if not sliding_enabled else set())
    reasons=[reason for key,reason in FAILURES.items() if key in active and checks[key] is not True]
    return dict(fit_preview_C_rec=not reasons,failure_reasons=reasons,gates=checks,
        certificate_event=None,formal_status='PENDING_FREEZE_AND_INDEPENDENT_CALIBRATION',
        disabled_gates=[k for k in FAILURES if k not in active])
