# -*- coding:utf-8 -*-
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import json

app = Flask(__name__)

model = joblib.load('models/test_model_0.model')

# External city and province data (Wikipedia)
city_stats = pd.read_csv('external_data/city_stats_wiki.csv')
city_stats.drop('Województwo', axis=1, inplace=True)
city_stats.columns = ['city', 'county', 'city_area', 'city_population', 'city_density']

province_stats = pd.read_csv('external_data/province_stats_wiki.csv')
province_stats.drop('Lp.', axis=1, inplace=True)
province_stats.columns = ['province', 'province_population', 'province_men_population', 'province_women_population']


def fe(df):
    """Main feature engineering function. Returned dataframe is ready to prediction."""
    
    def parse_location_city(val):
        """Using external data from wikipedia checks if value parsed from location feature is city."""
        all_city = city_stats['city'].to_list()
        for city_ in reversed(val):
        # "Józefów" apears more then one time on on all cities list and "Dobra" appears also as a street name.
        # I decided to exclude them, but it can be improved.
            if city_ in ['Dobra', 'Józefów']:
                continue
            if city_ in all_city:
                    return city_
        return 'other' 
    
    def normalize_build_year(year):
        """Normalize 'build_year' feature"""
        years = [1970, 1980, 1990, 2000, 2005, 2010, 2012, 2014, 2016, 2017]
        if year < 1970: return 1900
        if year > 2017: return 2018
    
        for idx in range(len(years) - 1):
            if years[idx+1] > year >= years[idx]:
                return years[idx]
    

    def normalize_floors_in_building(val):
        """Cap max floors in building number to control outliers"""
        floor = float(val)
        return floor if floor < 20 else 25
    
          
    def categorize_from_json(df, featname):
        """Import dictionary with label encoding used for training model
        and used it to categorize dataframe feature"""
        with open('model_predict_data/cat_dict_{}.txt'.format(featname), 'r', encoding='cp437',
                 errors='ignore') as file:
            cat_dict = json.load(file)
            
        return df[featname].map(cat_dict)

    
    def import_aggregated_df_from_csv(groupby_feats, feat):
        """Import aggregations used for training model"""        
        filename = 'groupby_{}_{}.csv'.format('_'.join(groupby_feats), feat)
        with open('model_predict_data/{}'.format(filename), 'r', encoding='cp437',
                 errors='ignore') as file:
            groupby_df = pd.read_csv(file)
        return groupby_df
    
  
    def is_primary_market_conc(df, feat):
        """Concatenate "is_primary_market" with other feature and categorize the results.
        Imported dictionary with category labels was used for training model"""
        df['is_primary_market_{}'.format(feat)] = df[ ['is_primary_market', feat] ].apply(
            lambda x: '{}_{}'.format(x['is_primary_market'], x[feat]), axis=1
        )

        df['is_primary_market_{}_cat'.format(feat)] = categorize_from_json(df, 'is_primary_market_{}'.format(feat))

        return df

    
    # Area
    df['area_num'] = df.area.astype(float)
    area_num_99 = np.percentile(df['area_num'], 99)
    df['area_norm'] = df['area_num'].map(lambda x: x if x <= area_num_99 else area_num_99)
    df['area_num_log'] = np.log(df['area_num'])    
    
    # Rooms
    df['area_per_room'] = df['area_norm'] / df["rooms"]    

    # Location 
    province_cities = ['Białystok', 'Bydgoszcz', 'Gdańsk', 'Gorzów Wielkopolski', 'Katowice', 'Kielce', 'Kraków', 'Lublin',
    'Łódź', 'Olsztyn', 'Opole', 'Poznań', 'Rzeszów', 'Szczecin', 'Toruń', 'Warszawa', 'Wrocław', 'Zielona Góra']
    
    df['province'] = df['location'].map(lambda x: x[0])
    df['city'] = df['location'].map(parse_location_city)
    df['province_city'] = df['city'].isin(province_cities)
    
    # Merging main dataframe with external data about cities.
    if 'city_area' not in df.columns:
        df = pd.merge(df, city_stats, on='city', how='left')
    # Merging main dataframe with external data about provinces.    
    if 'province_population' not in df.columns:
        df = pd.merge(df, province_stats, on='province', how='left')
        
    
    """'Location' feature is list cointaining elements describing property location in order
    from general to specyfic, which could be [<province>, <county>, <city>, <district> and <street>].
    """    
    for i in range(5):        
        # We can assume that "loc1" is likely province, "loc2" is likely county and so on.
        df["loc{}".format(i)] = df["location"].map(lambda x: x[i] if len(x) > i else "")      
    
    df['loc01'] = df['loc0'] + df['loc1']
    df['loc012'] = df['loc0'] + df['loc1'] + df['loc2']
    df['loc12'] = df['loc1'] + df['loc2']
    
    # Categorize location features
    for i in range(5):
        df["loc{}_cat".format(i)] = categorize_from_json(df, 'loc{}'.format(i))
    df["loc01_cat"] = categorize_from_json(df, 'loc01')
    df["loc012_cat"] = categorize_from_json(df, 'loc012')
    df["loc12_cat"] = categorize_from_json(df, 'loc12')
    
    df['city_cat'] = categorize_from_json(df, 'city')
    df['county_cat'] = categorize_from_json(df, 'county')
    df['province_cat'] = categorize_from_json(df, 'province')
  
    big_cities = {'Poznań', 'Sopot', 'Wrocław', 'Kraków', 'Gdańsk', 'Gdynia', 'Opole', 'Katowice',  'Częstochowa', 'Szczecin', 'Kalisz', 'Łódź', 'Olsztyn', 'Warszawa'}
    for city in big_cities:
        df[city] = df['city'] == city
        df['big_city'] = df['city'].map(lambda x: x in big_cities)
    
    # loc1 is likely to be "city", and loc2 is likely to be "district", so with combining this two
    # we could get for example: WrocławKrzyki, WarszawaŚródmieście, SopotGórny and so on.
    df_val_cnts = df['loc12'].value_counts()
    
    # We takes combinations only if they occur more then 100 times in dataset.
    loc12_vals = set(df_val_cnts[ df_val_cnts > 100].index.values)
    for item in loc12_vals:
        df[item] = df['loc12'] == item 
        
    # Floor
    floors_dict = {'parter': 0, '> 10': 11, 'poddasze': -2, 'suterena': -1}
    df['floor_num'] = df['floor'].map(lambda x: floors_dict.get(x, x)).fillna(-10).astype('int')
    
    # Floors_in_building
    df['floors_in_building_num'] = df['floors_in_building'].map(normalize_floors_in_building)
   
    # "price" aggregations    
    groupby_city_price = import_aggregated_df_from_csv(['city'], 'price')       
    if 'median_city_price' not in df:
        df = pd.merge(df, groupby_city_price, on='city', how='left')
        
    groupby_county_price = import_aggregated_df_from_csv(['county'], 'price')   
    if 'median_county_price' not in df:
        df = pd.merge(df, groupby_county_price, on='county', how='left')

    # is_primary_market
    df = is_primary_market_conc(df, 'rooms')
    df = is_primary_market_conc(df, 'city')
    df = is_primary_market_conc(df, 'rodzaj zabudowy') 
    
    # "price_m2" aggregations for concateneted is_primary_market with other features.   
    groupby_price_m2 = import_aggregated_df_from_csv(['is_primary_market_rooms'], 'price_m2')
    if 'median_is_primary_market_rooms_price_m2' not in df:
        df = pd.merge(df, groupby_price_m2, on='is_primary_market_rooms', how='left')
        
    groupby_price_m2 = import_aggregated_df_from_csv(['is_primary_market_rodzaj zabudowy'], 'price_m2')
    if 'median_is_primary_market_rodzaj zabudowy_price_m2' not in df:
        df = pd.merge(df, groupby_price_m2, on='is_primary_market_rodzaj zabudowy', how='left')
        
    
    # rok budowy            
    df['build_year'] = df['rok budowy'].fillna(-1).astype('int')   
    df["build_year_norm"] = df["build_year"].map(normalize_build_year)
    
    df['security'] = df['system alarmowy'] | df['rolety antywłamaniowe'] | df['drzwi / okna antywłamaniowe']
    
    cat_feats = {         
        "materiał budynku": "build_material_cat",
        "okna": "window_cat",
        "stan wykończenia": "property_completion_cat",
        "rodzaj zabudowy": "property_type_cat",
        "ogrzewanie": "property_heating_cat",
        "forma własności": "own_property_cat"
         }    
    
    for feat_name, feat_new_name in cat_feats.items():    
        df[feat_new_name] = categorize_from_json(df, feat_name)
      
        #OHE
        df_dummies = pd.get_dummies(df[feat_name])
        df_dummies.columns = ['{0}_{1}'.format(feat_new_name, x) for x in df_dummies.columns]
        df = pd.concat([df, df_dummies], axis=1)     
    

    print('Done')   
    return df

@app.route('/predict', methods=['POST'])
def get_forecast():
    """Run server. After POST data in specyfic format it returns predictions"""
    
    try:        
        df = pd.DataFrame(request.json)
        print(df)
        if 'location' in df:
            df['location'] = df['location'].map(lambda x: x.split(','))
   

        df_fe = fe(df)
    
        feats = ['rooms', 'czynsz', 'is_primary_market', 'system alarmowy', 'rolety antywłamaniowe', 'drzwi / okna antywłamaniowe', 'area_num', 'area_norm', 'area_num_log', 'area_per_room', 'province_city', 'city_area', 'city_population', 'city_density', 'province_population', 'province_men_population', 'province_women_population', 'loc0_cat', 'loc1_cat', 'loc2_cat', 'loc3_cat', 'loc4_cat', 'loc01_cat', 'loc012_cat', 'loc12_cat', 'city_cat', 'county_cat', 'province_cat', 'Poznań', 'big_city', 'Wrocław', 'Kraków', 'Kalisz', 'Gdynia', 'Opole', 'Szczecin', 'Gdańsk', 'Sopot', 'Częstochowa', 'Olsztyn', 'Łódź', 'Warszawa', 'Katowice', 'GdańskJasień', 'kołobrzeskiKołobrzeg', 'WarszawaMokotów', 'WarszawaBielany', 'ŁódźŚródmieście', 'WarszawaOchota', 'GdańskStare Miasto', 'KrakówNowa Huta', 'BydgoszczFordon', 'świdnickiŚwidnica', 'WarszawaBiałołęka', 'ToruńChełmińskie Przedmieście', 'GdańskMorena', 'BydgoszczBartodzieje', 'Zielona Góra', 'WarszawaWilanów', 'WrocławKrzyki', 'KrakówCzyżyny', 'SzczecinCentrum', 'PoznańWinogrady', 'PoznańRataje', 'KrakówPodgórze', 'gdańskiPruszcz Gdański', 'LublinCzuby', 'BydgoszczCentrum', 'BydgoszczSzwederowo', 'PoznańNaramowice', 'KrakówStare Miasto', 'KatowicePiotrowice', 'KrakówKazimierz', 'KatowiceOsiedle Tysiąclecia', 'stargardzkiStargard', 'BydgoszczWyżyny', 'KrakówPrądnik Czerwony', 'Rzeszów', 'WrocławStare Miasto', 'WarszawaPraga-Południe', 'lubińskiLubin', 'TychyŻwaków', 'KatowiceOsiedle Paderewskiego', 'tatrzańskiZakopane', 'KrakówWola Justowska', 'KatowiceBrynów', 'GdańskWrzeszcz', 'ŁódźGórna', 'wejherowskiRumia', 'WrocławFabryczna', 'KatowiceJózefowiec', 'KrakówGrzegórzki', 'WrocławKlecina', 'KatowiceDolina Trzech Stawów', 'LublinŚródmieście', 'WarszawaWola', 'głogowskiGłogów', 'RzeszówDrabinianka', 'LublinLSM', 'ŁódźPolesie', 'KrakówRuczaj', 'LublinWrotków', 'GdyniaŚródmieście', 'WarszawaBemowo', 'BydgoszczKapuściska', 'tczewskiTczew', 'wielickiWieliczka', 'PoznańGrunwald', 'KielceŚlichowice', 'KielceCentrum', 'KatowiceŚródmieście', 'KrakówDębniki', 'GdyniaOrłowo', 'PoznańNowe Miasto', 'KrakówŚródmieście', 'GdańskŚródmieście', 'SopotDolny', 'GdańskŁostowice', 'WarszawaPraga-Północ', 'RzeszówSłocina', 'WrocławPsie Pole', 'KrakówKrowodrza', 'GdańskPrzymorze', 'KrakówBronowice', 'PoznańWilda', 'ToruńMokre', 'ełckiEłk', 'ŁódźBałuty', 'wołomińskiZąbki', 'WrocławŚródmieście', 'WarszawaUrsynów', 'KrakówPrądnik Biały', 'KrakówBieżanów-Prokocim', 'WarszawaŚródmieście', 'KatowiceWełnowiec', 'floor_num', 'floors_in_building_num', 'mean_city_price', 'median_city_price', 'mean_county_price', 'median_county_price', 'is_primary_market_rooms_cat', 'is_primary_market_city_cat', 'is_primary_market_rodzaj zabudowy_cat', 'mean_is_primary_market_rooms_price_m2', 'median_is_primary_market_rooms_price_m2', 'mean_is_primary_market_rodzaj zabudowy_price_m2', 'median_is_primary_market_rodzaj zabudowy_price_m2', 'build_year', 'build_year_norm', 'security', 'build_material_cat', 'build_material_cat_beton', 'build_material_cat_beton komórkowy', 'build_material_cat_cegła', 'build_material_cat_drewno', 'build_material_cat_inne', 'build_material_cat_keramzyt', 'build_material_cat_pustak', 'build_material_cat_silikat', 'build_material_cat_wielka płyta', 'build_material_cat_żelbet', 'window_cat', 'window_cat_aluminiowe', 'window_cat_drewniane', 'window_cat_plastikowe', 'property_completion_cat', 'property_completion_cat_do remontu', 'property_completion_cat_do wykończenia', 'property_completion_cat_do zamieszkania', 'property_type_cat', 'property_type_cat_apartamentowiec', 'property_type_cat_blok', 'property_type_cat_dom wolnostojący', 'property_type_cat_kamienica', 'property_type_cat_loft', 'property_type_cat_plomba', 'property_type_cat_szeregowiec', 'property_heating_cat', 'property_heating_cat_elektryczne', 'property_heating_cat_gazowe', 'property_heating_cat_inne', 'property_heating_cat_kotłownia', 'property_heating_cat_miejskie', 'property_heating_cat_piece kaflowe', 'own_property_cat', 'own_property_cat_pełna własność', 'own_property_cat_spółdzielcze wł. z kw', 'own_property_cat_spółdzielcze własnościowe', 'own_property_cat_udział']
        X = df_fe[feats].values
        y_pred = np.exp(model.predict(X))
        
        return jsonify(prices=[float(x) for x in y_pred], status='ok')
    except Exception as e:
        print(e)
        return jsonify(message='something is going wrong', status='error')
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8051, debug=True)
