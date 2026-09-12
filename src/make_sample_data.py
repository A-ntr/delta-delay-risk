"""
Generates a small synthetic flight dataset with the same columns and the same
*kinds* of patterns as real BTS data (some routes/carriers/aircraft are
chronically worse, delays are more common in winter), so you can build and
sanity-check the feature pipeline before downloading the real multi-GB file.

This is a stand-in for real data ONLY. Don't draw airline-performance
conclusions from it — it's synthetic.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

CARRIERS = ["DL", "AA", "UA", "WN", "B6", "NK"]
AIRPORTS = ["ATL", "JFK", "ORD", "DFW", "LAX", "SEA", "MIA", "BOS", "DEN", "MSP"]
N_TAILS = 300


def make_sample(n_rows: int = 50_000, start_date: str = "2023-01-01",
                 end_date: str = "2024-12-31") -> pd.DataFrame:
    dates = pd.date_range(start_date, end_date, freq="D")
    tails = [f"N{100+i}DL" for i in range(N_TAILS)]

    # Give some tails/routes/carriers a hidden "chronic delay bias" so the
    # historical-rate features actually have signal to find.
    tail_bias = dict(zip(tails, RNG.normal(0, 0.08, size=N_TAILS)))
    route_bias = {}
    carrier_bias = dict(zip(CARRIERS, RNG.normal(0, 0.05, size=len(CARRIERS))))

    rows = []
    for _ in range(n_rows):
        date = RNG.choice(dates)
        month = pd.Timestamp(date).month
        origin, dest = RNG.choice(AIRPORTS, size=2, replace=False)
        route = f"{origin}-{dest}"
        route_bias.setdefault(route, RNG.normal(0, 0.06))
        carrier = RNG.choice(CARRIERS)
        tail = RNG.choice(tails)
        dep_hour = RNG.integers(5, 23)
        distance = RNG.integers(200, 2500)

        # winter months + red-eye/early departures nudge delay probability up
        seasonal = 0.06 if month in (12, 1, 2) else 0.0
        tod = 0.03 if dep_hour < 7 or dep_hour > 20 else 0.0

        base_p = 0.08 + tail_bias[tail] + route_bias[route] + carrier_bias[carrier] + seasonal + tod
        p_delay = float(np.clip(base_p, 0.01, 0.85))
        delayed = RNG.random() < p_delay
        arr_delay_minutes = max(0, RNG.normal(35, 15)) if delayed else max(0, RNG.normal(-2, 5))

        rows.append(dict(
            FlightDate=pd.Timestamp(date),
            Reporting_Airline=carrier,
            Origin=origin,
            Dest=dest,
            Tail_Number=tail,
            CRSDepTime=f"{dep_hour:02d}00",
            DayOfWeek=pd.Timestamp(date).dayofweek + 1,
            Distance=distance,
            ArrDelayMinutes=round(arr_delay_minutes, 1),
            Cancelled=0,
        ))

    return pd.DataFrame(rows).sort_values("FlightDate").reset_index(drop=True)


if __name__ == "__main__":
    df = make_sample()
    out_path = "data/raw/sample_flights.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} rows to {out_path}")
    print(df.head())
