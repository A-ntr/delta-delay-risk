# Getting the real data

TranStats (transtats.bts.gov) doesn't offer a simple direct-download URL — it's
a form-based portal, and it's not reachable from this sandboxed environment.
You'll need to pull it yourself on your own machine:

1. Go to https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ
   (Reporting Carrier On-Time Performance, 1987–present).
2. Pick a year and month range — start with **1 full year** to keep the file
   manageable (a single month is ~500k rows).
3. Under "Filter Year"/"Filter Period," select your range.
4. Check the fields you need at minimum:
   `FlightDate, Reporting_Airline, Origin, Dest, CRSDepTime, DayOfWeek,
   Distance, Tail_Number, ArrDelayMinutes, Cancelled`
5. Click Download — it exports a zipped CSV.
6. Unzip it into this folder (`data/raw/`).

For aircraft age, use the FAA Registry: https://registry.faa.gov/aircraftinquiry/
(N-Number search or bulk download) — join on `Tail_Number` to get the
manufacture/registration date.

## In the meantime
Run `python src/make_sample_data.py` to generate a small synthetic dataset with
the same schema and the same kinds of patterns (route/carrier/tail effects,
seasonal delay bumps) so you can build and test the pipeline logic locally
before wrestling with the real multi-GB file.
