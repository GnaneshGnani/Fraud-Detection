"""
quick_lb_check.py  ‒ Fire concurrent POST requests at an AWS ALB and
summarise how many responses came from each ECS task.

▪ Requires:  pip install requests
▪ Usage:     python quick_lb_check.py --url https://ALB-DNS/predict \
                                      --workers 20 --requests 300
"""
import argparse, json, random, string, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

PAYLOAD = {
    "features": [
        -0.260647805, -0.46964845,  2.496266083, -0.083723913, 0.129681236,
         0.73289825,  0.519013618, -0.130006048, 0.727159269, 0.637734541,
        -0.98702001,  0.2934381,   -0.941386125, 0.549019894, 1.804878578,
         0.215597994, 0.512306661, 0.333643717, 0.124270156, 0.091201899,
        -0.11055168,  0.217606144, -0.134794495, 0.165959115, 0.126279976,
        -0.434823981, -0.081230109, -0.151045486, 17982.1
    ]
}

# change this to whatever identifier your endpoint returns
KEY = "hostname"

def single_call(session, url):
    """Send one POST and return (identifier, full_response | error_msg)."""
    try:
        r = session.post(url, json=PAYLOAD, timeout=10)
        r.raise_for_status()
        data = r.json()
        ident = data.get(KEY, "unknown")
        return ident, data
    except Exception as exc:
        return "ERROR", str(exc)

def bombard(url: str, total: int, workers: int):
    counter, errors = Counter(), []
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as pool, requests.Session() as s:
        futures = [pool.submit(single_call, s, url) for _ in range(total)]
        for fut in as_completed(futures):
            ident, payload_or_err = fut.result()
            if ident == "ERROR":
                errors.append(payload_or_err)
            else:
                counter[ident] += 1

    elapsed = time.perf_counter() - t0
    return counter, errors, elapsed

def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument("--url", required=True, help="Full URL to POST")
    ap.add_argument("--requests", type=int, default=200,
                    help="Total number of requests (default 200)")
    ap.add_argument("--workers", type=int, default=20,
                    help="Concurrent threads (default 20)")
    args = ap.parse_args()

    cnt, errs, secs = bombard("http://Fraud-Detection-Load-Balancer-666632599.us-east-2.elb.amazonaws.com/predict", args.requests, args.workers)

    print(f"\nSent {args.requests} requests in {secs:.1f}s "
          f"({args.requests / secs:.1f} req/s).")

    if cnt:
        print("\nResponses by task:")
        for ident, n in cnt.most_common():
            pct = 100 * n / args.requests
            print(f"  {ident:<25} {n:>5}  ({pct:4.1f} %)")
    if errs:
        print(f"\n{len(errs)} errors:")
        for e in errs[:10]:
            print(" ", e)

if __name__ == "__main__":
    main()
