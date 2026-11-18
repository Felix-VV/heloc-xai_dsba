# Data

The FICO HELOC data are **not redistributed** with this code. Place the files as follows:

```
data/raw/heloc_dataset_v1.csv          # required
data/dictionary/HelocDataDict2.xlsx     # optional, for reference only (not read by the code)
```

## Official source

FICO released the data for the 2018 Explainable Machine Learning Challenge (FICO, Google and
academic partners) through the FICO Community site:
`https://community.fico.com/s/explainable-machine-learning-challenge`. Access there requires
registering and accepting FICO's terms; check those terms before using or sharing the data.
That page could not be reached from our build environment, so its current terms were **not**
checked by us.

## Raw file identity

- File: `heloc_dataset_v1.csv`, 10,459 data rows, 24 columns (`RiskPerformance` + 23 predictors),
  special codes -7/-8/-9 intact (this is the *raw* version, not a "cleaned" variant).
- SHA-256: `e5a914f742ad7f8472c5417902ec06b321ebc5de1d68ec13f136d62c87a7fbf3`
  (checked by `src/hxai/data.py`; the pipeline refuses another file).
- The following public mirrors provide the byte-identical raw file used in this study
  (same SHA-256):
  - https://raw.githubusercontent.com/yaronshap/GloballyConsistentRules/master/heloc_dataset_v1.csv
    (repository of Y. Shaposhnik, co-author of the FICO-challenge model by Chen et al.)
  - https://raw.githubusercontent.com/benoitparis/explainable-challenge/master/heloc_dataset_v1.csv
  - https://raw.githubusercontent.com/ckormylo/AI-Audit/master/heloc_dataset_v1.csv
  The same hash is also quoted independently in https://github.com/maitsingh/xai_heloc. Mirrors are not the
  official distribution channel; if you have the file from FICO, verify the hash.

## Data dictionary

The official FICO dictionary (`heloc_data_dictionary-2.xlsx`) was not reachable. We used a copy
redistributed for a university course:
https://raw.githubusercontent.com/ajzhanghk/Stat3612/master/HelocDataDict.xlsx
(SHA-256 `e13a37a8a8efdb22027e5df964a56994fb244085dfabda96052531f196543953`) and
https://raw.githubusercontent.com/ajzhanghk/Stat3612/master/HelocDataDict2.xlsx
(SHA-256 `810a73e2dba969635d7eb576b98d72ffdf042b09d47dbf8cabc880757876705f`). It renames variables x1-x23 but keeps FICO's descriptions, the
`MaxDelq` code tables, the special-value definitions and the monotonicity column:

| Code | Meaning (dictionary) |
|---|---|
| -9 | No Bureau Record or No Investigation |
| -8 | No Usable/Valid Trades or Inquiries |
| -7 | Condition not Met (e.g. No Inquiries, No Delinquencies) |

The dictionary labels the target "Paid as negotiated flag (12-36 Months)". The FICO challenge
description, as reproduced in IBM's AIX360 documentation, defines "Bad" as 90+ days past due at
least once over 24 months from account opening. We could not reconcile the two wordings against
an official source; the report states both.

Older "cleaned" copies (e.g. 9,871 rows with imputed values and the two `MaxDelq` fields removed)
have lost the special-code information and are not suitable for this pipeline.
