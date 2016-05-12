"""
Jinhui Wang
"""
    
import time
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta

def read_csv(filepath):
    
    #Read the csv files.
    department = pd.read_csv(filepath + 'department.csv')
    diagnosis = pd.read_csv(filepath + 'diagnosis.csv')
    medication_order = pd.read_csv(filepath + 'medication_order.csv')
    visit = pd.read_csv(filepath + 'visit.csv', parse_dates=['APPT_DT', 'APPT_CHECKIN_DT', 'APPT_MADE_DT', 'APPT_CHECKOUT_DT', 'HOSP_ADMIT_DT', 'HOSP_DISCHRG_DT'])
    visit_diagnosis = pd.read_csv(filepath + 'visit_diagnosis.csv')
    return department, diagnosis, medication_order, visit, visit_diagnosis

def cohort_construction(department, diagnosis, medication_order, visit, visit_diagnosis):
    
    visit = visit.ix[:, ['VISIT_KEY','PAT_KEY', 'DEPT_KEY', 'DICT_ENC_TYPE_KEY', 'AGE', 'HOSP_ADMIT_DT', 'HOSP_DISCHRG_DT']]
    
    #Patient's visit is a Hospital Encounter
    hospital_visit = visit[visit.DICT_ENC_TYPE_KEY == 83]
    
    #Hospital encounter occurs after "8/1/2014 00.00.00".
    date_string = '2014-08-01 00:00:00'
    hospital_encounter = hospital_visit[hospital_visit.HOSP_ADMIT_DT > datetime.datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')]

    #Patient's age is between 1 and 18
    patients_age = hospital_encounter[(hospital_encounter.AGE>=1.0) & (hospital_encounter.AGE<=18.0)]
    
    #Patient received an Emergency Department diagnosis (primary or secondary)  of anaphylaxis or allergic reaction, identified by the following ICD9 Codes
    icd9 = ['995.0', '995.3', '995.6', '995.60', '995.61', '995.62', '995.63', '995.64', '995.65', '995.66', '995.67', '995.68', '995.69', '995.7', '999.4', '999.41', '999.42', '999.49']
    icd9_filter_diag = diagnosis[diagnosis['ICD9_CD'].isin(icd9)]
    icd9_diag_index = icd9_filter_diag.ix[:, ['ICD9_CD', 'DX_KEY']]
    ed_diagnosis = visit_diagnosis[(visit_diagnosis['DICT_DX_STS_KEY'] == 313) | (visit_diagnosis['DICT_DX_STS_KEY'] == 314)]
    ed_diagnosis = ed_diagnosis.merge(icd9_diag_index, on='DX_KEY', how='inner')
    filtered_visit_diag = ed_diagnosis.ix[:, ['VISIT_KEY', 'DX_KEY']]
    cohort_ER_diag = patients_age.merge(filtered_visit_diag, on='VISIT_KEY', how = 'inner')
    
    #The encounter was not at an Urgent Care department
    department_urgent = department[department.SPECIALTY.str.contains('URGENT CARE', case=False)| department.DEPT_CNTR.str.contains('URGENT CARE', case=False)]
    urgent_care = department_urgent['DEPT_KEY'].tolist()
    
    #filter the urgent care department to get the final cohort
    cohort = cohort_ER_diag[~cohort_ER_diag['DEPT_KEY'].isin(urgent_care)]
    
    return cohort


def create_indicators(cohort, diagnosis, medication_order, visit):

    
    #Add ANAPH_DX_IND
    diag_anaphylaxis = diagnosis[diagnosis.DX_NM.str.contains('anaphylaxis', case=False)]
    diag_index = list(set(diag_anaphylaxis['DX_KEY'].tolist()))
    cohort['ANAPH_DX_IND'] = np.where(cohort['DX_KEY'].isin(diag_index), 1, 0)
    cohort_with_dx = cohort.groupby(['VISIT_KEY'])['ANAPH_DX_IND'].apply(lambda x: x.max())
    cohort_with_dx = cohort_with_dx.reset_index()
    temp = cohort.drop('ANAPH_DX_IND', axis=1)
    cohort_with_dx = temp.merge(cohort_with_dx,on='VISIT_KEY', how = 'left')
    
    #Add EPI_ORDER
    medication = medication_order.ix[:, ['PAT_KEY','MED_ORD_NM']]
    temp2 = temp.merge(medication, on='PAT_KEY', how = 'left')
    temp2['EPI_ORDER'] = np.where((temp2['MED_ORD_NM'].str.contains('epinephrine', case=False)) & (temp2['MED_ORD_NM'].notnull()), 1, 0)
    cohort_with_med = temp2.groupby(['PAT_KEY'])['EPI_ORDER'].apply(lambda x: x.max())
    cohort_with_med = cohort_with_med.reset_index()
    cohort_with_med = cohort_with_dx.merge(cohort_with_med, on='PAT_KEY', how='left')
    
   
    #Add FOLLOW_UP
    outpatient_visit = visit[visit.DICT_ENC_TYPE_KEY == 108]
    outpatient_visit = outpatient_visit.ix[:, ['PAT_KEY', 'APPT_CHECKIN_DT']]
    
    temp3 = temp.merge(outpatient_visit, on='PAT_KEY', how='left')
    temp3['FOLLOW_UP'] = np.where(((temp3['APPT_CHECKIN_DT'].dt.date-temp3['HOSP_DISCHRG_DT'].dt.date) <= timedelta(days=7)) & ((temp3['APPT_CHECKIN_DT'].dt.date-temp3['HOSP_DISCHRG_DT'].dt.date) >= timedelta(days=0)), 1, 0)
    cohort_follow_up = temp3.groupby(['PAT_KEY'])['FOLLOW_UP'].apply(lambda x: x.max())
    cohort_follow_up = cohort_follow_up.reset_index()
    cohort_follow_up = cohort_with_med.merge(cohort_follow_up, on='PAT_KEY', how='left')
    
    
    #Add FOLLOW_UP_DATE and DAYS_TO_FOLLOW_UP
    temp3['timediff'] = temp3['APPT_CHECKIN_DT'].dt.date-temp3['HOSP_DISCHRG_DT'].dt.date
    temp4 = temp3[(temp3['timediff'] >= timedelta(days=0)) & (temp3['timediff'] <= timedelta(days=7))]
    temp4 = temp4.loc[temp4.groupby('PAT_KEY')['timediff'].idxmin()]
    cohort_with_date = temp4.ix[:,['PAT_KEY', 'APPT_CHECKIN_DT', 'timediff']]
    cohort_with_date.ix[:,'timediff'] = cohort_with_date.ix[:,'timediff'].astype('timedelta64[D]').astype(int)
    cohort_with_date = cohort_follow_up.merge(cohort_with_date, on='PAT_KEY', how='left')
    cohort_with_date = cohort_with_date.rename(columns={'APPT_CHECKIN_DT': 'FOLLOW_UP_DATE', 'timediff': 'DAYS_TO_FOLLOW_UP'})
    final_df = cohort_with_date.drop_duplicates(['PAT_KEY', 'VISIT_KEY', 'HOSP_ADMIT_DT', 'AGE', 'ANAPH_DX_IND', 'EPI_ORDER', 'FOLLOW_UP', 'FOLLOW_UP_DATE', 'DAYS_TO_FOLLOW_UP'])
    
    return final_df


def main():
    
    filepath = './'
    department, diagnosis, medication_order, visit, visit_diagnosis = read_csv(filepath)
    cohort = cohort_construction(department, diagnosis, medication_order, visit, visit_diagnosis)
    final_df = create_indicators(cohort, diagnosis, medication_order, visit)
    final_df = final_df.ix[:,['PAT_KEY', 'VISIT_KEY', 'HOSP_ADMIT_DT', 'AGE', 'ANAPH_DX_IND', 'EPI_ORDER', 'FOLLOW_UP', 'FOLLOW_UP_DATE', 'DAYS_TO_FOLLOW_UP']]
    final_df = final_df.set_index(['PAT_KEY'])
    final_df.to_csv('Data_Exercise_output_dataset.csv', sep=',', encoding='utf-8')

    
if __name__ == "__main__":
    main()



