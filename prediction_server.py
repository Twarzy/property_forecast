# -*- coding:utf-8 -*-
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from prediction import fe  # Main feature-engineering function.

app = Flask(__name__)

model = joblib.load('models/test_model_0.model')

@app.route('/')
def main_page():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def get_forecast():
    """Run server. After POST data in specific format it returns predictions"""
    
    try:        
        df = pd.DataFrame(request.json)
        print(df)
        if 'location' in df:
            df['location'] = df['location'].map(lambda x: x.split(','))
   

        df_fe = fe(df)
    
        feats = ['rooms', 'czynsz', 'is_primary_market', 'system alarmowy', 'rolety antywłamaniowe', 'drzwi / okna antywłamaniowe', 'area_num', 'area_norm', 'area_num_log', 'area_per_room', 'province_city', 'city_area', 'city_population', 'city_density', 'province_population', 'province_men_population', 'province_women_population', 'loc0_cat', 'loc1_cat', 'loc2_cat', 'loc3_cat', 'loc4_cat', 'loc01_cat', 'loc012_cat', 'loc12_cat', 'city_cat', 'county_cat', 'province_cat', 'Poznań', 'big_city', 'Wrocław', 'Kraków', 'Kalisz', 'Gdynia', 'Opole', 'Szczecin', 'Gdańsk', 'Sopot', 'Częstochowa', 'Olsztyn', 'Łódź', 'Warszawa', 'Katowice', 'GdańskJasień', 'kołobrzeskiKołobrzeg', 'WarszawaMokotów', 'WarszawaBielany', 'ŁódźŚródmieście', 'WarszawaOchota', 'GdańskStare Miasto', 'KrakówNowa Huta', 'BydgoszczFordon', 'świdnickiŚwidnica', 'WarszawaBiałołęka', 'ToruńChełmińskie Przedmieście', 'GdańskMorena', 'BydgoszczBartodzieje', 'Zielona Góra', 'WarszawaWilanów', 'WrocławKrzyki', 'KrakówCzyżyny', 'SzczecinCentrum', 'PoznańWinogrady', 'PoznańRataje', 'KrakówPodgórze', 'gdańskiPruszcz Gdański', 'LublinCzuby', 'BydgoszczCentrum', 'BydgoszczSzwederowo', 'PoznańNaramowice', 'KrakówStare Miasto', 'KatowicePiotrowice', 'KrakówKazimierz', 'KatowiceOsiedle Tysiąclecia', 'stargardzkiStargard', 'BydgoszczWyżyny', 'KrakówPrądnik Czerwony', 'Rzeszów', 'WrocławStare Miasto', 'WarszawaPraga-Południe', 'lubińskiLubin', 'TychyŻwaków', 'KatowiceOsiedle Paderewskiego', 'tatrzańskiZakopane', 'KrakówWola Justowska', 'KatowiceBrynów', 'GdańskWrzeszcz', 'ŁódźGórna', 'wejherowskiRumia', 'WrocławFabryczna', 'KatowiceJózefowiec', 'KrakówGrzegórzki', 'WrocławKlecina', 'KatowiceDolina Trzech Stawów', 'LublinŚródmieście', 'WarszawaWola', 'głogowskiGłogów', 'RzeszówDrabinianka', 'LublinLSM', 'ŁódźPolesie', 'KrakówRuczaj', 'LublinWrotków', 'GdyniaŚródmieście', 'WarszawaBemowo', 'BydgoszczKapuściska', 'tczewskiTczew', 'wielickiWieliczka', 'PoznańGrunwald', 'KielceŚlichowice', 'KielceCentrum', 'KatowiceŚródmieście', 'KrakówDębniki', 'GdyniaOrłowo', 'PoznańNowe Miasto', 'KrakówŚródmieście', 'GdańskŚródmieście', 'SopotDolny', 'GdańskŁostowice', 'WarszawaPraga-Północ', 'RzeszówSłocina', 'WrocławPsie Pole', 'KrakówKrowodrza', 'GdańskPrzymorze', 'KrakówBronowice', 'PoznańWilda', 'ToruńMokre', 'ełckiEłk', 'ŁódźBałuty', 'wołomińskiZąbki', 'WrocławŚródmieście', 'WarszawaUrsynów', 'KrakówPrądnik Biały', 'KrakówBieżanów-Prokocim', 'WarszawaŚródmieście', 'KatowiceWełnowiec', 'floor_num', 'floors_in_building_num', 'mean_city_price', 'median_city_price', 'mean_county_price', 'median_county_price', 'is_primary_market_rooms_cat', 'is_primary_market_city_cat', 'is_primary_market_rodzaj zabudowy_cat', 'mean_is_primary_market_rooms_price_m2', 'median_is_primary_market_rooms_price_m2', 'mean_is_primary_market_rodzaj zabudowy_price_m2', 'median_is_primary_market_rodzaj zabudowy_price_m2', 'build_year', 'build_year_norm', 'security', 'build_material_cat', 'build_material_cat_beton', 'build_material_cat_beton komórkowy', 'build_material_cat_cegła', 'build_material_cat_drewno', 'build_material_cat_inne', 'build_material_cat_keramzyt', 'build_material_cat_pustak', 'build_material_cat_silikat', 'build_material_cat_wielka płyta', 'build_material_cat_żelbet', 'window_cat', 'window_cat_aluminiowe', 'window_cat_drewniane', 'window_cat_plastikowe', 'property_completion_cat', 'property_completion_cat_do remontu', 'property_completion_cat_do wykończenia', 'property_completion_cat_do zamieszkania', 'property_type_cat', 'property_type_cat_apartamentowiec', 'property_type_cat_blok', 'property_type_cat_dom wolnostojący', 'property_type_cat_kamienica', 'property_type_cat_loft', 'property_type_cat_plomba', 'property_type_cat_szeregowiec', 'property_heating_cat', 'property_heating_cat_elektryczne', 'property_heating_cat_gazowe', 'property_heating_cat_inne', 'property_heating_cat_kotłownia', 'property_heating_cat_miejskie', 'property_heating_cat_piece kaflowe', 'own_property_cat', 'own_property_cat_pełna własność', 'own_property_cat_spółdzielcze wł. z kw', 'own_property_cat_spółdzielcze własnościowe', 'own_property_cat_udział']
        
        # Temporary bug fix - start
        for feat in feats:
            if feat not in df_fe.columns:
                df_fe[feat] = 0
        # Temporary bug fix - end
        
        X = df_fe[feats].values
        y_pred = np.exp(model.predict(X))
        
        return jsonify(prices=[float(x) for x in y_pred], status='ok')
    except Exception as e:
        print(e)
        return jsonify(message='something is going wrong', status='error')
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8051, debug=True)
