
"""
HospitalData.py
Examples in Jupyter Notebook to plot hospitalization data within MA from '20200702' to '20210109' using HospitalData.py and joined_file_20210114.csv being within same folder
--------

"""

import os
import pandas as pd #import turicreate as tc
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


import csv
import requests
import urllib

class HospitalData(object):
    ''' Represents a Turicreate SFrame of csv data extracted from HHS (+ covidtracking.com) hospitalization data
    https://healthdata.gov/dataset/covid-19-reported-patient-impact-and-hospital-capacity-state-timeseries
    https://api.covidtracking.com/v1/states/daily.csv

    Attributes
    ----------
    data: turicreate SFrame
        Sframe containing columns such as date, state, GeneralWard, OnVentICU, OffVentICU... full dataset, no filters
    filtered_data: turicreate SFrame
        Sframe containing columns such as date, state, GeneralWard, OnVentICU, OffVentICU... filtered dataset based on attr us_state, start_date, end_date
    csv_filename : str
        filepath to csv that was loaded/saved to initialize this HospitalData object
    us_state : str
        2 letter abbreviation for state of interest when using methods get_GeneralWard_counts(), get_OnVentICU_counts(), get_OffVentICU_counts()
    start_date : str
        YYYYMMDD str format of the start date of interest when using methods get_GeneralWard_counts(), get_OnVentICU_counts(), get_OffVentICU_counts()
    end_date : str
        YYYYMMDD str format of the end date of interest when using methods get_GeneralWard_counts(), get_OnVentICU_counts(), get_OffVentICU_counts()
    '''

    def __init__(self, us_state, start_date, end_training_date, end_date, population_data_obj = None):
        ''' Construct a HospitalData from provided input
        Args
        ----
        csv_filename : str
            filepath to csv that was loaded/saved to initialize this HospitalData object
        us_state : str
        2 letter abbreviation for state of interest when using methods get_GeneralWard_counts(), get_OnVentICU_counts(), get_OffVentICU_counts()
        start_date : str
            YYYYMMDD str format of the start date of interest when using methods get_GeneralWard_counts(), get_OnVentICU_counts(), get_OffVentICU_counts()
        end_date : str
            YYYYMMDD str format of the end date of interest when using methods get_GeneralWard_counts(), get_OnVentICU_counts(), get_OffVentICU_counts()
        Returns
        -------
        Newly constructed HospitalData instance
        '''
        self.data = pd.DataFrame([]) #tc.SFrame([])
        self.filtered_data = pd.DataFrame([]) #tc.SFrame([]) 
        self._us_state = us_state # protected attrib
        self._start_date = start_date # protected attrib
        self._end_training_date = end_training_date # protected attrib
        self._end_date = end_date # protected attrib
        self.population_data_obj = population_data_obj


        self.load_data()

    def get_HHS_data(self):
        ## download csv from hhs website to the current directory of this HospitalData.py file
        today = datetime.now().date()
        csv_name = 'reported_hospital_utilization_timeseries_%s.csv' % str(today).replace('-', '')

        if not Path(csv_name).exists() or datetime.fromtimestamp(os.path.getctime(csv_name)).date() != today:
            print('DOWNLOADING FRESH HHS DATA.........')

            ## OLD VERSION TO DOWNLOAD HHS DATA, DOES NOT WORK ANYMORE
            # from bs4 import BeautifulSoup
            # hhs_url = 'https://healthdata.gov/dataset/covid-19-reported-patient-impact-and-hospital-capacity-state-timeseries'
            # u = urllib.request.urlopen(hhs_url)
            # content_decoded = u.read().decode('utf-8')
            # parsed_html = BeautifulSoup(content_decoded)
            # url = parsed_html.body.find('a', attrs ={'class':'data-link'})['href']
            # response = requests.get(url)  
            # with open(csv_name, 'w') as f:
            #     writer = csv.writer(f)
            #     for line in response.iter_lines():
            #         writer.writerow(line.decode('utf-8').split(','))

            # return pd.read_csv(csv_name) #tc.SFrame(csv_name) # raw to be handled by join_HHS_covidtracking_data
            
            ## NEW VERSION, snopped copied and adapted from https://dev.socrata.com/foundry/healthdata.gov/g62h-syeh
            from sodapy import Socrata
            # Unauthenticated client only works with public data sets. Note 'None'
            # in place of application token, and no username or password:
            client = Socrata("healthdata.gov", None)

            # Example authenticated client (needed for non-public datasets):
            # client = Socrata(healthdata.gov,
            #                  MyAppToken,
            #                  userame="user@example.com",
            #                  password="AFakePassword")

            # First N results, returned as JSON from API / converted to Python list of
            # dictionaries by sodapy.
            results = client.get("g62h-syeh", limit=50000) # download all possible results (can't figure out how to download *last* N results)

            # Convert to pandas DataFrame
            results_df = pd.DataFrame.from_records(results)
            results_df.to_csv(csv_name)

            return pd.read_csv(csv_name)
        else: 
            return pd.read_csv(csv_name) #tc.SFrame(csv_name)
    def get_UMN_data(self):
        ## download csv from covidtracking website to the current directory of this HospitalData.py file
        today = datetime.now().date()
        csv_name = 'umn_%s.csv' % str(today).replace('-', '')
        
        if not Path(csv_name).exists() or datetime.fromtimestamp(os.path.getctime(csv_name)).date() != today:
            print('DOWNLOADING FRESH UMN DATA.........')
            url = 'https://csom-mili-api-covidhospitalizations.s3.us-east-2.amazonaws.com/hospitalizations/hospitalizations.csv'
            response = requests.get(url)  
            with open(csv_name, 'w') as f:
                writer = csv.writer(f)
                for line in response.iter_lines():
                    writer.writerow(line.decode('utf-8').split(','))
            tc_data = pd.read_csv(csv_name) #tc.SFrame(csv_name)
            print(tc_data)
            tc_data['date'] = tc_data.apply(lambda x: x['Date'].replace('-',''), axis=1)
            tc_data['state'] = tc_data.apply(lambda x: x['StateAbbreviation'], axis=1)

            # [print(x) for x in tc_data['CurrentlyInICU']]

            tc_data['inIcuCurrently'] = tc_data['CurrentlyInICU'] #tc_data.apply(lambda x: float('nan') if x['CurrentlyInICU'] == '' else int(float(x['CurrentlyInICU'])))
            tc_data['onVentilatorCurrently'] = tc_data.apply(lambda x: x['CurrentlyOnVentilator'], axis=1)
            # tc_data['deathIncrease'] = tc_data.apply(lambda x: x['TotalDeaths'])


            # def fill_deathIncrease():
            # unique_states = tc_data['state'].unique()
            # state_deaths_tcdict = {k:tc_data.filter_by([k],'state').sort('date',ascending=True) for k in unique_states}
            
            # for s in unique_states:
            #     sf = state_deaths_tcdict[s]                
            # tc_data['deathIncrease'] = tc_data.apply(lambda x: x['TotalDeaths'] - \
            #     state_deaths_tcdict[x['state']].filter_by([ datetime.strftime(datetime.strptime(x['date'],'%Y%m%d')-timedelta(days=1),'%Y%m%d') ],'date')['TotalDeaths'] )
            
            tc_data.to_csv(csv_name) #tc_data.save(csv_name, format='csv') 
            
            # return tc_data
            return pd.read_csv(csv_name)
        else:
#             print('get_covidtracking_data',tc.SFrame(csv_name))
            return pd.read_csv(csv_name)

    def join_HHS_UMN_data(self, hhs_data, umn_data):
        print(umn_data)
    
        ## joining the relevant column names of the 2 data sources
        
        # column name meanings found here: https://healthdata.gov/covid-19-reported-patient-impact-and-hospital-capacity-state-data-dictionary
        columns_of_interest_hhs = ['previous_day_admission_adult_covid_confirmed', \
         'total_adult_patients_hospitalized_confirmed_covid','staffed_icu_adult_patients_confirmed_covid']
        columns_of_interest_hhs = columns_of_interest_hhs+ [col + '_coverage' for col in columns_of_interest_hhs] # get the num of hospitals that reports each columns_of_interest
        columns_of_interest_hhs = ['date','state']+columns_of_interest_hhs
        hhs_data_truncated = hhs_data.loc[:, columns_of_interest_hhs] #hhs_data_truncated = hhs_data.select_columns(columns_of_interest_hhs)

        # columns_of_interest_ctrack = ['date','state','inIcuCurrently', 'onVentilatorCurrently','deathIncrease']
        columns_of_interest_umn = ['date','state','inIcuCurrently', 'onVentilatorCurrently','TotalDeaths']
        umn_data_truncated = umn_data.loc[:, columns_of_interest_umn] #ctrack_data_truncated = ctrack_data.select_columns(columns_of_interest_ctrack)

        # for both of the truncated Sframe, create a state_and_date column such that we can do a joining of rows (wherever they share the same state and same date)
        hhs_data_truncated['state_and_date'] = hhs_data_truncated.apply(lambda x: x['state'] + x['date'].split('T')[0].replace("-", ""), axis=1) #for example: "MA" + "2020-01-01".replace("-","")
        umn_data_truncated['state_and_date'] = umn_data_truncated.apply(lambda x: x['state'] + str(x['date']), axis=1) #for example: "MA" + str(20200101)
        # delete the 'date' and 'state' column of hhs_data_truncated because we will be getting those 2 columns from  ctrack_data_truncated after join
        hhs_data_truncated = hhs_data_truncated.drop(columns=['state','date']) #hhs_data_truncated = hhs_data_truncated.remove_columns(column_names=['state','date'])

        # hhs_data_truncated['state_and_date'] = hhs_data_truncated['state_and_date'].astype(str)
        # ctrack_data_truncated['state_and_date'] = ctrack_data_truncated['state_and_date'].astype(str)

        return hhs_data_truncated.merge(umn_data_truncated, on=['state_and_date'], how='left')

    def load_data(self):
        self.data = self.join_HHS_UMN_data(self.get_HHS_data(),self.get_UMN_data())
        
        self.filtered_data = self.get_filtered_data()
        

    def get_filtered_data(self):
        # returns Sframe within specified [self.start_date, self.end_date] and self.us_state 
        self.data['selected_row'] = self.data.apply(lambda x: 1 if (x['state']==self.us_state and (x['date']>=int(self._start_date) and x['date']<=int(self._end_date)) ) else 0, axis=1) #1 means row selected, 0 means row not selected for filtered data
        filtered_data =  self.data[self.data['selected_row'] == 1].sort_values(['date'], ascending=True) #self.data.filter_by([1],'selected_row').sort(['date'],ascending=True)
    
        TotalDeaths = filtered_data['TotalDeaths'].to_numpy()
        
        # print('-------DEBUGGG--------')
        # print(filtered_data['TotalDeaths'])
        # print(filtered_data['TotalDeaths'].to_numpy()[0])
        # print(filtered_data['TotalDeaths'].to_numpy()[:-1])
        TotalDeaths_1 = np.append(filtered_data['TotalDeaths'].to_numpy()[0],filtered_data['TotalDeaths'].to_numpy()[:-1])

        #calculate deathIncrease from (daily) TotalDeaths
        filtered_data['deathIncrease'] = TotalDeaths-TotalDeaths_1

        return filtered_data

    def get_InGeneralWard_counts(self):
        # returns InGeneralWard counts within [self.start_date, self.end_date] of the self.us_state specified
        return (self.filtered_data.apply(lambda x: x['total_adult_patients_hospitalized_confirmed_covid']-x['inIcuCurrently'] if (np.isnan(x['total_adult_patients_hospitalized_confirmed_covid']) or np.isnan(x['staffed_icu_adult_patients_confirmed_covid'])) \
            else x['total_adult_patients_hospitalized_confirmed_covid']-x['staffed_icu_adult_patients_confirmed_covid'], axis=1)).to_numpy()

    def get_ICU_counts(self):
        # returns total icu counts within [self.start_date, self.end_date] of the self.us_state specified
        return (self.filtered_data.apply(lambda x: x['staffed_icu_adult_patients_confirmed_covid'] if np.isnan(x['inIcuCurrently']) \
            else x['inIcuCurrently'], axis=1)).to_numpy()

    def get_Death_counts(self):
        # returns death counts within [self.start_date, self.end_date] of the self.us_state specified
        return (self.filtered_data.apply(lambda x: float('nan') if np.isnan(x['deathIncrease']) \
            else x['deathIncrease'], axis=1)).to_numpy()

    def get_Admission_counts(self):
        # returns admission counts within [self.start_date, self.end_date] of the self.us_state specified
        return (self.filtered_data.apply(lambda x: float('nan') if np.isnan(x['previous_day_admission_adult_covid_confirmed']) \
            else x['previous_day_admission_adult_covid_confirmed'], axis=1)).to_numpy()

    def get_OnVentInICU_counts(self):
    # # returns OnVentInICU counts within [self.start_date, self.end_date] of the self.us_state specified
        return (self.filtered_data.apply(lambda x: float('nan') if np.isnan(x['onVentilatorCurrently']) \
        else x['onVentilatorCurrently'], axis=1)).to_numpy()

    def get_OffVentInICU_counts(self):
    # # returns OffVentInICU counts within [self.start_date, self.end_date] of the self.us_state specified
        return (self.filtered_data.apply(lambda x: x['staffed_icu_adult_patients_confirmed_covid'] - x['onVentilatorCurrently'] if (np.isnan(x['inIcuCurrently']) or np.isnan(x['onVentilatorCurrently'])) \
        else (x['inIcuCurrently'] - x['onVentilatorCurrently']), axis=1)).to_numpy()



    ######fcns useful for ParamsGenerator to automatically make params.json file##########


    def get_init_num(self, hospital_state, template_params=None): #hospital state means Presenting,InGeneralWard,OffVentInICU,OnVentInICU,etc
        return self.get_warm_start_init2(hospital_state, template_params)


    def get_warm_start_init2(self, hospital_state, template_params):
        day0 = self.start_date
        warm_up_dict = dict()
        pre_start_horizon = -template_params['num_past_timesteps']

        warm_up_horizon = 17 #  give a warm_up_horizon #days for the model to warm up each hospital state counts

        labeled_GeneralWard = self.get_InGeneralWard_counts()[0:warm_up_horizon]
        labeled_ICU = self.get_ICU_counts()[0:warm_up_horizon]
        labeled_ICU_OnVent = self.get_OnVentInICU_counts()[0:warm_up_horizon] 
        labeled_ICU_OffVent = self.get_OffVentInICU_counts()[0:warm_up_horizon]
        labeled_Admissions = self.get_Admission_counts()[0:warm_up_horizon]
        horizon_counts_dict = {'InGeneralWard':labeled_GeneralWard, 'ICU':labeled_ICU,'OffVentInICU':labeled_ICU_OffVent,'OnVentInICU':labeled_ICU_OnVent,'Admissions':labeled_Admissions}
        day0_count = (horizon_counts_dict[hospital_state][0])
        day0_count_50 = 0.50*day0_count
        day0_count_200 = 2*day0_count

        if hospital_state == 'InGeneralWard':
            for p in range(pre_start_horizon,1,1):
                warm_up_dict[str(p)] = int(day0_count_200/10) 
            return warm_up_dict


        if hospital_state == 'OnVentInICU':
            for p in range(pre_start_horizon,1,1):
                warm_days = 5
                if p>=-warm_days:
                    warm_up_dict[str(p)] = int(day0_count_50/warm_days) 
                else:
                    warm_up_dict[str(p)] = int(0)
            return warm_up_dict

        else:  #hospital_state == 'OffVentInICU'
            return int(0)

    def get_num_timesteps_in_days(self):
        # "num_timesteps": 21,
        dateFormat = "%Y%m%d"
        d1 = datetime.strptime(self._start_date, dateFormat)
        d2 = datetime.strptime(self._end_date, dateFormat)
        return abs((d2 - d1).days)


    def get_pmf_num_per_timestep_InGeneralWard(self, ):
        # returns a filename.csv of pmf_num_per_timestep_Presenting
        previous_day_admission_adult_covid_confirmed = self.filtered_data.apply(lambda x: float('nan') if x['previous_day_admission_adult_covid_confirmed']==None else x['previous_day_admission_adult_covid_confirmed'], axis=1).to_numpy()

        available_hhs_admission_dates = self.filtered_data.apply(lambda x: float('nan') if x['previous_day_admission_adult_covid_confirmed']==None else str(x['date']), axis=1)
        if self.population_data_obj is None: # NOT using Population Model defined in Josh Cohen workplan
            presenting_per_day = np.append(previous_day_admission_adult_covid_confirmed[1:],np.array(0)) # to keep length consistent
        else: # append population model forecasts from Josh Cohen workplan
            missing_hhs_admissions_dates = []
            for t in range((datetime.strptime((self.population_data_obj.dates_to_forecast[0]), "%Y%m%d") - datetime.strptime(available_hhs_admission_dates[-1], "%Y%m%d")).days ):
                missing_hhs_admissions_dates += [int((datetime.strptime(available_hhs_admission_dates[-1], "%Y%m%d") +timedelta(days=t)).strftime("%Y%m%d"))]
            print('missing_hhs_admissions_dates', missing_hhs_admissions_dates)
            imputed_missing_hhs_admissions = self.population_data_obj.filtered_data[self.population_data_obj.filtered_data['date'] == missing_hhs_admissions_dates].sort_values(['date'])['severe'].to_numpy()*self.population_data_obj.params['prob_hosp'] #imputed_missing_hhs_admissions = self.population_data_obj.filtered_data.filter_by(missing_hhs_admissions_dates, 'date').sort(['date'])['severe'].to_numpy()*self.population_data_obj.params['prob_hosp']

            # DEBUG MODE FOR checking if pop data forecasting working
            if self.population_data_obj.debug_mode:
                previous_day_admission_adult_covid_confirmed[1:] = 0
                #isolates forecasting data for debugging by setting all other existent hosp-admission data to 0

            presenting_per_day = np.append(np.append(previous_day_admission_adult_covid_confirmed[1:],\
                                    np.array(imputed_missing_hhs_admissions)), \
                                    np.array(self.population_data_obj.forecasted_data['hosp'].to_numpy()))
            

            print(len(previous_day_admission_adult_covid_confirmed[1:]), len(imputed_missing_hhs_admissions), len(self.population_data_obj.forecasted_data['hosp'].to_numpy()), 'lens within admissions.csv DEBUGGG')
            print((self.filtered_data['date'].unique().sort_values()), 'dates of unique filtered data')
            print((self.population_data_obj.dates_to_forecast), 'dates to forecast DEBUGGG')
        pmf_SFrame = pd.DataFrame({'timestep':list(range(len(presenting_per_day))),'num_InGeneralWard':presenting_per_day}) #pmf_SFrame = tc.SFrame({'timestep':list(range(len(presenting_per_day))),'num_InGeneralWard':presenting_per_day})
        csv_filename = str(Path(__file__).resolve().parents[0])+ '/'+self.us_state+'_'+self.start_date+'_'+self.end_date +'_'+'pmf_num_per_timestep_InGeneralWard.csv' 
        pmf_SFrame.to_csv(csv_filename) #pmf_SFrame.save(csv_filename, format='csv')
        return csv_filename

    # fcn called within ParamsGenerator class
    def extract_param_value(self,param_name, template_params=None):
        # function that maps param_name to param_value based on the supplemental_obj being HospitalData or CovidEstim Obj
        states = ["InGeneralWard", "OffVentInICU", "OnVentInICU"]
        
        if 'num_timesteps' == param_name:
            return self.get_num_timesteps_in_days()

        if 'pmf_num_per_timestep_InGeneralWard' == param_name:
            return self.get_pmf_num_per_timestep_InGeneralWard()

        for hospital_state in states: 
            if 'init_num_'+hospital_state == param_name:
                return self.get_init_num(hospital_state,template_params)
        
        # if does not hit any of above conditional cases
        raise Exception(param_name+' failed compatibility with extract_param_value of :'+type(self))
    ######END fcns useful for ParamsGenerator##########

    @property
    def start_date(self):
        return self._start_date

    @property
    def end_date(self):
        return self._end_date

    @property
    def us_state(self):
        return self._us_state

    # if a user decides to update any of the protected state, then HospitalData.filtered_data should correspondingly be updated 
    @start_date.setter
    def start_date(self, new_value):
        self._start_date = new_value
        self.filtered_data = self.get_filtered_data()

    @end_date.setter
    def end_date(self, new_value):
        self._end_date = new_value
        self.filtered_data = self.get_filtered_data()

    @us_state.setter
    def us_state(self, new_value):
        self._us_state = new_value
        self.filtered_data = self.get_filtered_data()

