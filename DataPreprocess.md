# Preprocessing Data

## Libraries Used
- pandas
- pickle
- numpy
- matplotlib

### Step 1: Import Data

Use columns: 
- `PATIENT_ID`	
- `period`
- `decease`
- `decompensation`	
- `Lactate dehydrogenase`	
- `High sensitivity C-reactive protein`	
- `lymphocyte count`

### Step 2: Refine DATA

- `total_period = Discharge time - Admission Time`
- keep interval as `3` and divide `total_period` by `interval`
- `period` as `Discharge time - RE_DATE` and apply `lambda` function by dividing `period` by `interval`
- set `period` as `total_period - period`
- group rows by `PATIENT_ID` and `period` and reset index
- change `outcome` to `decease`
- create column `decompensation` to check if the patient dies in the next 24 hours
- drop columns
    - RE_DATE
    - outcome
    - Admission Time
    - Discharge Time
    - total_period
- convert data types to float
- use columns
    - 'PATIENT_ID'
    - 'period'
    - 'decease'
    - 'decompensation'
    - 'Lactate dehydrogenase'
    - 'High sensitivity' 
    - 'C-reactive protein'
    - 'lymphocyte count'