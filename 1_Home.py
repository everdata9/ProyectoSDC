import streamlit as st
import pandas as pd
import requests
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
from mplsoccer import Radar,FontManager,grid,Pitch, Sbopen
from highlight_text import fig_text
from mplsoccer import PyPizza, FontManager
from matplotlib import rcParams
import numpy as np
from PIL import Image


text_contents = '''PJ-Partidos jugados /
                       GLS-Goles /
                       xG-Goles esperados /
                       Sh-Remates /
                       PasCompT-Pases completados totales /
                       Tkl-Número total de takles /
                       Ast-Asistencias /
                       PrgP-Pases progresivos /
                       PrgC-Carreras progresivas /
                       Min-Minutos /
                       PK-Penales /
                       TA-Tarjetas amarillas /
                       TR-Tarjetas rojas /
                       SoT-Tiros directos /
                       G/SH-Goles por tiro /
                       G/SoT-Goles por tiro directo /
                       FK-Tiros libres /
                       PasCort-Pases cortos /
                       PasLarg-Pases largos /
                       DistSh-Distancia promedio de tiro /
                       TklAtt-Takles en ataque /
                       TklMed-Takles en el medio /
                       TklDef-Takels en defensa /
                       PasUltTer-Pases en zona ofensiva'''    
txt_Position='''DF - Defensores /
                    MF - Centrocampistas /
                    FW - Delanteros /
                    FB - Laterales /
                    LB - Laterales izquierdos /
                    RB - Laterales derechos /
                    CB - Defensas centrales /
                    DM - Centrocampistas defensivos /
                    CM - Centrocampistas centrales /
                    LM - Centrocampistas izquierdos /
                    RM - Centrocampistas derechos /
                    WM - Centrocampistas anchos /
                    LW - Ala izquierda /
                    RW - Ala derecha /
                    AM - Centrocampistas ofensivos''' 
fechaAct='📅 Fecha de actualización: 03/05/2023'

def GetSimilarPlayers(PlayerName, numPlayers, corr_matrix):
    
    SimPlayers = pd.DataFrame(columns = ['PlayerName', 'Similar Player', 'Correlation Factor'])

    i = 0
    for i in range(0, numPlayers):
        row = corr_matrix.loc[corr_matrix.index == PlayerName].squeeze()

        SimPlayers.at[i, 'PlayerName'] = PlayerName
        SimPlayers.at[i, 'Similar Player'] = row.nlargest(i+2).sort_values(ascending=True).index[0]
        SimPlayers.at[i, 'Correlation Factor'] = row.nlargest(i+2).sort_values(ascending=True)[0]

        i = i+1
    
    return SimPlayers
def GetMexicanData():
    
    # Cargamos el dataset (en la carpeta data con nombre PCA)
    dt_standar = pd.read_csv('Data/mexico_standard_22_23.csv', sep = ';')
    dt_shooting = pd.read_csv('Data/mexico_shooting_22_23.csv', sep = ';')
    dt_passing = pd.read_csv('Data/mexico_passing_22_23.csv', sep = ';')
    dt_defensive = pd.read_csv('Data/mexico_defensive_22_23.csv', sep = ';')
    df_mex_detalles = pd.merge(dt_standar,dt_shooting,left_on = 'Rk',right_on='Rk').merge(dt_passing,left_on = 'Rk',right_on='Rk').merge(dt_defensive,left_on = 'Rk',right_on='Rk')

    columns={"Player": "Player","Pos":"Pos","Age": "Age","Squad":"Squad","MP": "PJ", "Min": "Min", "Gls_x": "Gls","Ast_x": "Ast","PK_x":"PK","CrdY":"TA","CrdR":"TR","xG_x":"xG","xAG_x":"xAG","PrgC":"PrgC","PrgP_x":"PrgP","PrgR":"PrgR", #Standar
         "Sh_x":"Sh","SoT":"SoT","G/Sh":"G/Sh","G/SoT":"G/SoT","Dist":"DistSh","FK":"FK",                                                                                   #Shooting
         "Cmp":"PasCompT","Cmp%.1":"PasCort","Cmp%.3":"PasLarg","1/3":"PasUltTer",                                                                                          #Passing
         "Tkl":"Tkl","Def 3rd":"TklDef","Mid 3rd":"TklMed","Att 3rd":"TklAtt","Int":"Int"
         } 

    claves = list(columns.values())

    df_mex_detalles.rename(
    columns=columns,
    inplace=True,
    )

    df_mex_detalles=df_mex_detalles[claves]
    df_mex_detalles.fillna(0,inplace=True)
   
    ##indexNames = df_mex_detalles[(df_mex_detalles['Player'] == 'Luis Félix')].index
    #df_mex_detalles = df_mex_detalles.drop(df_mex_detalles[(df_mex_detalles['Player'] == 'Luis Félix')].index)
    ##df_mex_detalles.drop(indexNames , inplace=True)
    ##st.write(len(df_mex_detalles))
    #st.write(df_mex_detalles[(df_mex_detalles['Player'] == 'Emiliano García')])
    ##st.write(df_mex_detalles.loc[df_mex_detalles.duplicated(subset=['Player','Age'])])
    #######################################################################
    
    for index, row in df_mex_detalles.loc[df_mex_detalles.duplicated(subset=['Player','Age'])].iterrows():
        
        df_EquipoActual = df_mex_detalles.iloc[[index]]
        jugador = df_EquipoActual['Player'][index]
        age = df_EquipoActual['Age'][index]
        squad = df_EquipoActual['Squad'][index]
        df_EquipoAnterior = df_mex_detalles[(df_mex_detalles['Player'] == jugador) & 
                                (df_mex_detalles['Age'] == age) ]
                                #& 
                                #(df_mex_detalles['Squad'] != squad)]
        

        df_mex_detalles.loc[index, 'PJ'] = df_EquipoActual['PJ'][index] + df_EquipoAnterior['PJ'].values[0] 
        df_mex_detalles.loc[index, 'Min'] = df_EquipoActual['Min'][index] + df_EquipoAnterior['Min'].values[0]
        df_mex_detalles.loc[index, 'Gls'] = df_EquipoActual['Gls'][index] + df_EquipoAnterior['Gls'].values[0]
        df_mex_detalles.loc[index, 'Ast'] = df_EquipoActual['Ast'][index] + df_EquipoAnterior['Ast'].values[0]
        df_mex_detalles.loc[index, 'PK'] = df_EquipoActual['PK'][index] + df_EquipoAnterior['PK'].values[0]
        df_mex_detalles.loc[index, 'TA'] = df_EquipoActual['TA'][index] + df_EquipoAnterior['TA'].values[0]
        df_mex_detalles.loc[index, 'TR'] = df_EquipoActual['TR'][index] + df_EquipoAnterior['TR'].values[0]
        df_mex_detalles.loc[index, 'xG'] = df_EquipoActual['xG'][index] + df_EquipoAnterior['xG'].values[0]
        df_mex_detalles.loc[index, 'xAG'] = df_EquipoActual['xAG'][index] + df_EquipoAnterior['xAG'].values[0]
        df_mex_detalles.loc[index, 'PrgC'] = df_EquipoActual['PrgC'][index] + df_EquipoAnterior['PrgC'].values[0]
        df_mex_detalles.loc[index, 'PrgP'] = df_EquipoActual['PrgP'][index] + df_EquipoAnterior['PrgP'].values[0]
        df_mex_detalles.loc[index, 'PrgR'] = df_EquipoActual['PrgR'][index] + df_EquipoAnterior['PrgR'].values[0]
        df_mex_detalles.loc[index, 'Sh'] = df_EquipoActual['Sh'][index] + df_EquipoAnterior['Sh'].values[0]
        df_mex_detalles.loc[index, 'SoT'] = df_EquipoActual['SoT'][index] + df_EquipoAnterior['SoT'].values[0]
        df_mex_detalles.loc[index, 'G/Sh'] = (df_EquipoActual['G/Sh'][index] + df_EquipoAnterior['G/Sh'].values[0])/2
        df_mex_detalles.loc[index, 'G/SoT'] = (df_EquipoActual['G/SoT'][index] + df_EquipoAnterior['G/SoT'].values[0])/2
        df_mex_detalles.loc[index, 'DistSh'] = (df_EquipoActual['DistSh'][index] + df_EquipoAnterior['DistSh'].values[0])/2
        df_mex_detalles.loc[index, 'FK'] = df_EquipoActual['FK'][index] + df_EquipoAnterior['FK'].values[0]
        df_mex_detalles.loc[index, 'PasCompT'] = df_EquipoActual['PasCompT'][index] + df_EquipoAnterior['PasCompT'].values[0]
        df_mex_detalles.loc[index, 'PasCort'] = df_EquipoActual['PasCort'][index] + df_EquipoAnterior['PasCort'].values[0]
        df_mex_detalles.loc[index, 'PasLarg'] = df_EquipoActual['PasLarg'][index] + df_EquipoAnterior['PasLarg'].values[0]
        df_mex_detalles.loc[index, 'PasUltTer'] = df_EquipoActual['PasUltTer'][index] + df_EquipoAnterior['PasUltTer'].values[0]
        df_mex_detalles.loc[index, 'Tkl'] = df_EquipoActual['Tkl'][index] + df_EquipoAnterior['Tkl'].values[0]
        df_mex_detalles.loc[index, 'TklDef'] = df_EquipoActual['TklDef'][index] + df_EquipoAnterior['TklDef'].values[0]
        df_mex_detalles.loc[index, 'TklMed'] = df_EquipoActual['TklMed'][index] + df_EquipoAnterior['TklMed'].values[0]
        df_mex_detalles.loc[index, 'TklAtt'] = df_EquipoActual['TklAtt'][index] + df_EquipoAnterior['TklAtt'].values[0]
        df_mex_detalles.loc[index, 'Int'] = df_EquipoActual['Int'][index] + df_EquipoAnterior['Int'].values[0]
    
    df_mex_detalles = df_mex_detalles.drop_duplicates(subset=['Player','Age'], keep="last")
    df_mex_detalles['Player']=df_mex_detalles[['Player','Squad']].apply(lambda fila: str(fila[0]) + " (" +  str(fila[1] + ")"), axis=1)

    df_mex_detalles = df_mex_detalles.drop(["Age", "Squad"], axis=1)   
    return df_mex_detalles
def add_trunc(df):     
    params= list(df.columns)                  
    for x in params:
        df[x]=df[x].apply(lambda x: str(round((x),1)))
        df[x]=df[x].apply(lambda x: str(x).replace('.0',''))

    return df



st.set_page_config(
    page_title="everdata9",
    page_icon="🏠",
    layout="wide"
)


with st.sidebar:
    selected = option_menu("Menú Principal", ["Inicio",'Rendimiento Jugadores','Jugadores Similares','Rendimiento Equipos','Equipos Similares','Los Ángeles FC'], 
        icons=['house','lightning','diagram-3-fill','bar-chart-line','distribute-vertical','lightning'], menu_icon="cast", default_index=0)
    selected


if selected == "Inicio":
    
    st.title('Bienvenido a @everdata9 📈')
    st.markdown('<div style="text-align: justify; border: 2px solid gray; padding: 10px; font-size: 20px;"><i>@everdata9 es una herramienta de análisis pública y gratuita que pone a disposición técnicas de analítica de datos, machine learning e inteligencia artificial (IA) para los DELANTEROS profesionales de la Liga MX (Primera División de México)</i></div>', unsafe_allow_html=True)  
    st.write('')  
    st.write('')  
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        image = Image.open('./image/atlas.png')
        st.image(image)
        image = Image.open('./image/tigres.png')
        st.image(image)
        #st.write("")
        image = Image.open('./image/pachuca.png')
        st.image(image)
        #st.write("")
        image = Image.open('./image/juarez.png')
        st.image(image)
    with col2:
        image = Image.open('./image/america.png')
        st.image(image)
        #st.write("")
        image = Image.open('./image/tijuana.png')
        st.image(image)
        #st.write("")
        image = Image.open('./image/necaxa.png')
        st.image(image)
        #st.write("")
        image = Image.open('./image/mazatlan.png')
        st.image(image)
    with col3:
        image = Image.open('./image/atletico.png')
        st.image(image)
        #st.write("")
        image = Image.open('./image/santos.png')
        st.image(image)
        #st.write("")
        image = Image.open('./image/monterrey.png')
        st.image(image)
        #st.write("")
        image = Image.open('./image/toluca.png')
        st.image(image)
    with col4:
        image = Image.open('./image/azul.png')
        st.image(image)
        #st.write("")
        image = Image.open('./image/queretano.png')
        st.image(image)
        #st.write("")
        image = Image.open('./image/leon.png')
        st.image(image)
    with col5:
        image = Image.open('./image/pumas.png')
        st.image(image)
        #st.write("")
        image = Image.open('./image/puebla.png')
        st.image(image)
        #st.write("")
        image = Image.open('./image/chivas.png')
        st.image(image)
    st.write("")
    st.write(fechaAct) 
elif selected == "Rendimiento Jugadores":

    df= GetMexicanData().copy()

    col1, col2 = st.columns(2)


    with col1:

        expander = st.sidebar.expander("Seleccionar Jugadores")
        with expander:
            position_option= df['Pos'].unique().tolist()
            position_name=st.selectbox('Seleccionar posición',position_option,4)        
            if position_name=='FW':
                df_colums=df[['PJ','Min','Gls','Ast','PK','TA','TR','xG','xAG','PrgC','PrgP','PrgR','Sh','SoT','G/Sh','G/SoT','DistSh','FK','PasCompT','PasCort','PasLarg','PasUltTer','Tkl','TklDef','TklMed','TklAtt','Int']]
                df_colums_s=df[['PJ','Gls','xG','Sh','PasCompT','Tkl','Ast','PrgP','PrgC']]
                player_option= df[(df['Pos']=='FW')]['Player'].unique().tolist()
                metric_opction=df_colums.columns.tolist()
                metric_opction_s=df_colums_s.columns.tolist()
                players_name=st.multiselect('Seleccionar jugador',player_option,['Eduardo Aguirre (Santos)','Martín Barragán (Puebla)'])             

            elif position_name=='DF':
                df_colums=df[['PJ','Min','Gls','Ast','PK','TA','TR','xG','xAG','PrgC','PrgP','PrgR','Sh','SoT','G/Sh','G/SoT','DistSh','FK','PasCompT','PasCort','PasLarg','PasUltTer','Tkl','TklDef','TklMed','TklAtt','Int']]
                df_colums_s=df[['PJ','Gls','xG','Sh','PasCompT','Tkl','Ast','PrgP','PrgC']]
                player_option= df[(df['Pos']=='DF')]['Player'].unique().tolist()
                metric_opction=df_colums.columns.tolist()
                metric_opction_s=df_colums_s.columns.tolist()
                players_name=st.multiselect('Seleccionar jugador',player_option,['Luis Abram (Cruz Azul)','Gaddi Aguirre (Atlas)'])

            elif position_name=='MF':
                df_colums=df[['PJ','Min','Gls','Ast','PK','TA','TR','xG','xAG','PrgC','PrgP','PrgR','Sh','SoT','G/Sh','G/SoT','DistSh','FK','PasCompT','PasCort','PasLarg','PasUltTer','Tkl','TklDef','TklMed','TklAtt','Int']]
                df_colums_s=df[['PJ','Gls','xG','Sh','PasCompT','Tkl','Ast','PrgP','PrgC']]
                player_option= df[(df['Pos']=='MF')]['Player'].unique().tolist()
                metric_opction=df_colums.columns.tolist()
                metric_opction_s=df_colums_s.columns.tolist()
                players_name=st.multiselect('Seleccionar jugador',player_option,['Clifford Aboagye (Querétaro)','Pedro Aquino (América)'])

            elif position_name=='MFDF':
                df_colums=df[['PJ','Min','Gls','Ast','PK','TA','TR','xG','xAG','PrgC','PrgP','PrgR','Sh','SoT','G/Sh','G/SoT','DistSh','FK','PasCompT','PasCort','PasLarg','PasUltTer','Tkl','TklDef','TklMed','TklAtt','Int']]
                df_colums_s=df[['PJ','Gls','xG','Sh','PasCompT','Tkl','Ast','PrgP','PrgC']]
                player_option= df[(df['Pos']=='MFDF')]['Player'].unique().tolist()
                metric_opction=df_colums.columns.tolist()
                metric_opction_s=df_colums_s.columns.tolist()
                players_name=st.multiselect('Seleccionar jugador',player_option,['Fernando Illescas (Mazatlán)','Ignacio Rivero (Cruz Azul)'])

            elif position_name=='FWMF':
                df_colums=df[['PJ','Min','Gls','Ast','PK','TA','TR','xG','xAG','PrgC','PrgP','PrgR','Sh','SoT','G/Sh','G/SoT','DistSh','FK','PasCompT','PasCort','PasLarg','PasUltTer','Tkl','TklDef','TklMed','TklAtt','Int']]
                df_colums_s=df[['PJ','Gls','xG','Sh','PasCompT','Tkl','Ast','PrgP','PrgC']]
                player_option= df[(df['Pos']=='FWMF')]['Player'].unique().tolist()
                metric_opction=df_colums.columns.tolist()
                metric_opction_s=df_colums_s.columns.tolist()
                players_name=st.multiselect('Seleccionar jugador',player_option,['Isaác Brizuela (Guadalajara)','Damián Batallini (Necaxa)'])
             
            elif position_name=='DFMF':
                df_colums=df[['PJ','Min','Gls','Ast','PK','TA','TR','xG','xAG','PrgC','PrgP','PrgR','Sh','SoT','G/Sh','G/SoT','DistSh','FK','PasCompT','PasCort','PasLarg','PasUltTer','Tkl','TklDef','TklMed','TklAtt','Int']]
                df_colums_s=df[['PJ','Gls','xG','Sh','PasCompT','Tkl','Ast','PrgP','PrgC']]
                player_option= df[(df['Pos']=='DFMF')]['Player'].unique().tolist()
                metric_opction=df_colums.columns.tolist()
                metric_opction_s=df_colums_s.columns.tolist()
                players_name=st.multiselect('Seleccionar jugador',player_option,['José Lozano (Santos)','Alek Álvarez (UNAM)'])
            
            elif position_name=='DFFW':
                df_colums=df[['PJ','Min','Gls','Ast','PK','TA','TR','xG','xAG','PrgC','PrgP','PrgR','Sh','SoT','G/Sh','G/SoT','DistSh','FK','PasCompT','PasCort','PasLarg','PasUltTer','Tkl','TklDef','TklMed','TklAtt','Int']]
                df_colums_s=df[['PJ','Gls','xG','Sh','PasCompT','Tkl','Ast','PrgP','PrgC']]
                player_option= df[(df['Pos']=='DFFW')]['Player'].unique().tolist()
                metric_opction=df_colums.columns.tolist()
                metric_opction_s=df_colums_s.columns.tolist()
                players_name=st.multiselect('Seleccionar jugador',player_option,['Carlos Gutiérrez (UNAM)','Eduardo Tercero (UANL)'])

            elif position_name=='MFFW':
                df_colums=df[['PJ','Min','Gls','Ast','PK','TA','TR','xG','xAG','PrgC','PrgP','PrgR','Sh','SoT','G/Sh','G/SoT','DistSh','FK','PasCompT','PasCort','PasLarg','PasUltTer','Tkl','TklDef','TklMed','TklAtt','Int']]
                df_colums_s=df[['PJ','Gls','xG','Sh','PasCompT','Tkl','Ast','PrgP','PrgC']]
                player_option= df[(df['Pos']=='MFFW')]['Player'].unique().tolist()
                metric_opction=df_colums.columns.tolist()
                metric_opction_s=df_colums_s.columns.tolist()
                players_name=st.multiselect('Seleccionar jugador',player_option,['Pablo Barrera (Querétaro)','Nicolás Benedetti (Mazatlán)'])           

            elif position_name=='FWDF':
                df_colums=df[['PJ','Min','Gls','Ast','PK','TA','TR','xG','xAG','PrgC','PrgP','PrgR','Sh','SoT','G/Sh','G/SoT','DistSh','FK','PasCompT','PasCort','PasLarg','PasUltTer','Tkl','TklDef','TklMed','TklAtt','Int']]
                df_colums_s=df[['PJ','Gls','xG','Sh','PasCompT','Tkl','Ast','PrgP','PrgC']]
                player_option= df[(df['Pos']=='FWDF')]['Player'].unique().tolist()
                metric_opction=df_colums.columns.tolist()
                metric_opction_s=df_colums_s.columns.tolist()
                players_name=st.multiselect('Seleccionar jugador',player_option,['Jacobo Reyes (Monterrey)','Juan Domínguez (Necaxa)'])           


            players_name=sorted(players_name)
            colums=st.multiselect('Seleccionar métricas',metric_opction,metric_opction_s)
            #colums.insert(0,'Squad')
            colums.insert(0,'Player')

            x = 0
            cadena_nombre=""
            while x < len(players_name):
                cadena_nombre=cadena_nombre+"<"+players_name[x]+">"+" | "
                x += 1

        df_Forward= df[colums].copy()

        if len(players_name)==0:
            mask= (df_Forward['Player']=='Julián Quiñones (Atlas)')|(df_Forward['Player']=='Alexis Vega (Guadalajara)')
        elif len(players_name)==1:
            mask= (df_Forward['Player']==players_name[0])
        elif len(players_name)==2:
            mask= (df_Forward['Player']==players_name[0])|(df_Forward['Player']==players_name[1])
        elif len(players_name)==3:
            mask= (df_Forward['Player']==players_name[0])|(df_Forward['Player']==players_name[1])|(df_Forward['Player']==players_name[2])
        elif len(players_name)==4:
            mask= (df_Forward['Player']==players_name[0])|(df_Forward['Player']==players_name[1])|(df_Forward['Player']==players_name[2])|(df_Forward['Player']==players_name[3])
        elif len(players_name)==5:
            mask= (df_Forward['Player']==players_name[0])|(df_Forward['Player']==players_name[1])|(df_Forward['Player']==players_name[2])|(df_Forward['Player']==players_name[3])|(df_Forward['Player']==players_name[4])


        df_Forward=df_Forward[mask].reset_index()
        df_Forward=df_Forward.sort_values(by=['Player'])
        df_Forward=df_Forward.drop_duplicates(subset="Player",)

        if len(players_name)==0:
            a_values=df_Forward.iloc[0].values.tolist()    
            b_values=df_Forward.iloc[1].values.tolist()    
            a_values=a_values[3:]
            b_values=b_values[3:]

        elif len(players_name)==1:
            a_values=df_Forward.iloc[0].values.tolist()      
            a_values=a_values[3:]

        elif len(players_name)==2:
            a_values=df_Forward.iloc[0].values.tolist()    
            b_values=df_Forward.iloc[1].values.tolist()    
            a_values=a_values[3:]
            b_values=b_values[3:]

        elif len(players_name)==3:
            a_values=df_Forward.iloc[0].values.tolist()    
            b_values=df_Forward.iloc[1].values.tolist()    
            c_values=df_Forward.iloc[2].values.tolist()    
            a_values=a_values[3:]
            b_values=b_values[3:]
            c_values=c_values[3:]

        elif len(players_name)==4:
            a_values=df_Forward.iloc[0].values.tolist()    
            b_values=df_Forward.iloc[1].values.tolist()    
            c_values=df_Forward.iloc[2].values.tolist()    
            d_values=df_Forward.iloc[3].values.tolist()    
            a_values=a_values[3:]
            b_values=b_values[3:]
            c_values=c_values[3:]   
            d_values=d_values[3:]  

        elif len(players_name)==5:
            a_values=df_Forward.iloc[0].values.tolist()    
            b_values=df_Forward.iloc[1].values.tolist()    
            c_values=df_Forward.iloc[2].values.tolist()    
            d_values=df_Forward.iloc[3].values.tolist()    
            e_values=df_Forward.iloc[4].values.tolist()   
            a_values=a_values[3:]
            b_values=b_values[3:]
            c_values=c_values[3:]   
            d_values=d_values[3:]  
            e_values=e_values[3:]  

        params= list(df_Forward.columns)

        params= params[3:]
        #add ranges to list
        ranges=[]
        low = [] 
        high =[] 
        for x in params:
            a=min(df_Forward[params][x])
            a= a - (a * .25)

            b=max(df_Forward[params][x])
            b=b-(b*.25)     

            if a==b:
                b=b+0000.1

            low.append((a))
            high.append((b))

        # Add anything to this list where having a lower number is better
        # this flips the statistic
        lower_is_better = ['Miscontrol']

        radar = Radar(params, low, high,
                lower_is_better=lower_is_better,
                # whether to round any of the labels to integers instead of decimal places
                round_int=[False]*len(params),
                num_rings=4,  # the number of concentric circles (excluding center circle)
                # if the ring_width is more than the center_circle_radius then
                # the center circle radius will be wider than the width of the concentric circles
                ring_width=1, center_circle_radius=1)


        URL2 = ('https://raw.githubusercontent.com/googlefonts/SourceSerifProGFVersion/main/fonts/'
                'SourceSerifPro-ExtraLight.ttf')
        serif_extra_light = FontManager(URL2)
        URL3 = ('https://raw.githubusercontent.com/google/fonts/main/ofl/rubikmonoone/'
                'RubikMonoOne-Regular.ttf')
        rubik_regular = FontManager(URL3)
        URL4 = 'https://raw.githubusercontent.com/googlefonts/roboto/main/src/hinted/Roboto-Thin.ttf'
        robotto_thin = FontManager(URL4)
        URL5 = ('https://raw.githubusercontent.com/google/fonts/main/apache/robotoslab/'
                'RobotoSlab%5Bwght%5D.ttf')
        robotto_bold = FontManager(URL5)

        # plot radar
        fig, ax = radar.setup_axis()
        rings_inner = radar.draw_circles(ax=ax, facecolor='#ffb2b2', edgecolor='#fc5f5f')

        if len(players_name)==0:      
            radar1, vertices1 = radar.draw_radar_solid(a_values, ax=ax,
                                                    kwargs={'facecolor': '#aa65b2',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#502a54',
                                                            'lw': 3})

            radar2, vertices2 = radar.draw_radar_solid(b_values, ax=ax,
                                                    kwargs={'facecolor': '#66d8ba',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#216352',
                                                            'lw': 3})

            ax.scatter(vertices1[:, 0], vertices1[:, 1],
                    c='#aa65b2', edgecolors='#502a54', marker='o', s=150, zorder=2)
            ax.scatter(vertices2[:, 0], vertices2[:, 1],
                    c='#66d8ba', edgecolors='#216352', marker='o', s=150, zorder=2)   

        elif len(players_name)==1:      
            radar1, vertices1 = radar.draw_radar_solid(a_values, ax=ax,
                                                    kwargs={'facecolor': '#aa65b2',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#216352',
                                                            'lw': 3})

            ax.scatter(vertices1[:, 0], vertices1[:, 1],
                    c='#aa65b2', edgecolors='#502a54', marker='o', s=150, zorder=2)

        elif len(players_name)==2:      
            radar1, vertices1 = radar.draw_radar_solid(a_values, ax=ax,
                                                    kwargs={'facecolor': '#aa65b2',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#502a54',
                                                            'lw': 3})

            radar2, vertices2 = radar.draw_radar_solid(b_values, ax=ax,
                                                    kwargs={'facecolor': '#66d8ba',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#216352',
                                                            'lw': 3})

            ax.scatter(vertices1[:, 0], vertices1[:, 1],
                    c='#aa65b2', edgecolors='#502a54', marker='o', s=150, zorder=2)
            ax.scatter(vertices2[:, 0], vertices2[:, 1],
                    c='#66d8ba', edgecolors='#216352', marker='o', s=150, zorder=2)
        
        elif len(players_name)==3:      
            radar1, vertices1 = radar.draw_radar_solid(a_values, ax=ax,
                                                    kwargs={'facecolor': '#aa65b2',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#216354',
                                                            'lw': 3})

            radar2, vertices2 = radar.draw_radar_solid(b_values, ax=ax,
                                                    kwargs={'facecolor': '#66d8ba',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#216352',
                                                            'lw': 3})

            radar3, vertices3 = radar.draw_radar_solid(c_values, ax=ax,
                                                    kwargs={'facecolor': '#697cd4',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#222b54',
                                                            'lw': 3})     
            ax.scatter(vertices1[:, 0], vertices1[:, 1],
                    c='#aa65b2', edgecolors='#502a54', marker='o', s=150, zorder=2)
            ax.scatter(vertices2[:, 0], vertices2[:, 1],
                    c='#66d8ba', edgecolors='#216352', marker='o', s=150, zorder=2)
            ax.scatter(vertices3[:, 0], vertices3[:, 1],
                    c='#697cd4', edgecolors='#222b54', marker='o', s=150, zorder=2)                                                                                             

        elif len(players_name)==4:      
            radar1, vertices1 = radar.draw_radar_solid(a_values, ax=ax,
                                                    kwargs={'facecolor': '#aa65b2',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#216354',
                                                            'lw': 3})

            radar2, vertices2 = radar.draw_radar_solid(b_values, ax=ax,
                                                    kwargs={'facecolor': '#66d8ba',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#216352',
                                                            'lw': 3})

            radar3, vertices3 = radar.draw_radar_solid(c_values, ax=ax,
                                                    kwargs={'facecolor': '#697cd4',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#222b54',
                                                            'lw': 3})
            radar4, vertices4 = radar.draw_radar_solid(d_values, ax=ax,
                                                    kwargs={'facecolor': '#778821',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#4D5B05',
                                                            'lw': 3})   
                                                                                                                    
            ax.scatter(vertices1[:, 0], vertices1[:, 1],
                    c='#aa65b2', edgecolors='#502a54', marker='o', s=150, zorder=2)
            ax.scatter(vertices2[:, 0], vertices2[:, 1],
                    c='#66d8ba', edgecolors='#216352', marker='o', s=150, zorder=2)
            ax.scatter(vertices3[:, 0], vertices3[:, 1],
                    c='#697cd4', edgecolors='#222b54', marker='o', s=150, zorder=2)  
            ax.scatter(vertices4[:, 0], vertices4[:, 1],
                    c='#778821', edgecolors='#4D5B05', marker='o', s=150, zorder=2)      

        elif len(players_name)==5:      
            radar1, vertices1 = radar.draw_radar_solid(a_values, ax=ax,
                                                    kwargs={'facecolor': '#aa65b2',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#216354',
                                                            'lw': 3})

            radar2, vertices2 = radar.draw_radar_solid(b_values, ax=ax,
                                                    kwargs={'facecolor': '#66d8ba',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#216352',
                                                            'lw': 3})

            radar3, vertices3 = radar.draw_radar_solid(c_values, ax=ax,
                                                    kwargs={'facecolor': '#697cd4',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#222b54',
                                                            'lw': 3})
            radar4, vertices4 = radar.draw_radar_solid(d_values, ax=ax,
                                                    kwargs={'facecolor': '#778821',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#4D5B05',
                                                            'lw': 3})   
            radar5, vertices5 = radar.draw_radar_solid(e_values, ax=ax,
                                                    kwargs={'facecolor': '#a82228',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#930e14',
                                                            'lw': 3})   

            ax.scatter(vertices1[:, 0], vertices1[:, 1],
                    c='#aa65b2', edgecolors='#502a54', marker='o', s=150, zorder=2)
            ax.scatter(vertices2[:, 0], vertices2[:, 1],
                    c='#66d8ba', edgecolors='#216352', marker='o', s=150, zorder=2)
            ax.scatter(vertices3[:, 0], vertices3[:, 1],
                    c='#697cd4', edgecolors='#222b54', marker='o', s=150, zorder=2)  
            ax.scatter(vertices4[:, 0], vertices4[:, 1],
                    c='#778821', edgecolors='#4D5B05', marker='o', s=150, zorder=2)   
            ax.scatter(vertices5[:, 0], vertices5[:, 1],
                    c='#a82228', edgecolors='#930e14', marker='o', s=150, zorder=2)   

        range_labels = radar.draw_range_labels(ax=ax, fontsize=20, fontproperties=robotto_bold.prop, color="#000000")
        param_labels = radar.draw_param_labels(ax=ax, fontsize=20, fontproperties=robotto_bold.prop, color="#000000")

        if len(players_name)==1:  
            fig_text(
                0.515, 0.99, cadena_nombre, size=22, fig=fig,
                highlight_textprops=[{"color": '#aa65b2'}],                                                                
                ha="center", fontproperties=robotto_bold.prop, color="#000000"
            )
        elif len(players_name)==2:  
                    fig_text(
                0.515, 0.99, cadena_nombre, size=22, fig=fig,
                highlight_textprops=[{"color": '#aa65b2'}, {"color": '#66d8ba'}],                                                                
                ha="center", fontproperties=robotto_bold.prop, color="#000000"
            )
        elif len(players_name)==3:  
                    fig_text(
                0.515, 0.99, cadena_nombre, size=22, fig=fig,
                highlight_textprops=[{"color": '#aa65b2'}, {"color": '#66d8ba'}, {"color": '#697cd4'}],                                                                
                ha="center", fontproperties=robotto_bold.prop, color="#000000"
            )
        elif len(players_name)==4:  
                    fig_text(
                0.515, 0.99, cadena_nombre, size=22, fig=fig,
                highlight_textprops=[{"color": '#aa65b2'}, {"color": '#66d8ba'}, {"color": '#697cd4'}, {"color": '#778821'}],                                                                
                ha="center", fontproperties=robotto_bold.prop, color="#000000"
            )
        elif len(players_name)==5:  
                    fig_text(
                0.515, 0.99, cadena_nombre, size=22, fig=fig,
                highlight_textprops=[{"color": '#aa65b2'}, {"color": '#66d8ba'}, {"color": '#697cd4'}, {"color": '#778821'}, {"color": '#a82228'}],                                                                
                ha="center", fontproperties=robotto_bold.prop, color="#000000"
            )
    
        # add subtitle
        fig.text(
            0.515, 0.942,
            "Liga MX (Primera División de México) | Temporada 2022-23",
            size=20,
            ha="center", fontproperties=robotto_bold.prop, color="#000000"
        )



            # add credits
        CREDIT_1 = "data: vía fbref"
        CREDIT_2 = "inspired by: @everdata9"

        fig.text(
            0.99, 0.005, f"{CREDIT_1}\n{CREDIT_2}", size=12,
            fontproperties=robotto_bold.prop, color="#000000",
            ha="right"
        )
        

        st.write(fig)

    with col2:     

        st.header('****Tabla con datos comparativos****')       
        
        df1_transposed=df_Forward.iloc[:, 1:len(df_Forward.columns)]        
        df1_transposed.set_index('Player', inplace=True)
        df1_transposed = df1_transposed.T 
        df1_transposed = add_trunc(df1_transposed)
        st.table(df1_transposed)
    
       # CSS to inject contained in a string
        hide_table_row_index = """
                    <style>
                        table {
                        border-collapse: collapse;
                        width: 100%;
                        }

                        th, td {
                        text-align: left;
                        padding: 8px;
                        }

                        tr:nth-child(even) {
                        background-color: #f4f4f4;
                        }
                    </style>
                    """
    st.write(fechaAct)
    st.header('MÉTRICAS')
    st.markdown(text_contents)
    st.header('POSICIONES')
    st.markdown(txt_Position)
elif selected == "Jugadores Similares":

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import plotly.graph_objects as go
    # Cargamos el dataset (en la carpeta data con nombre PCA)--
    dt_standar = pd.read_csv('Data/mexico_standard_22_23.csv', sep = ';')
    dt_shooting = pd.read_csv('Data/mexico_shooting_22_23.csv', sep = ';')
    dt_passing = pd.read_csv('Data/mexico_passing_22_23.csv', sep = ';')
    dt_defensive = pd.read_csv('Data/mexico_defensive_22_23.csv', sep = ';')
    df_mex_detalles = pd.merge(dt_standar,dt_shooting,left_on = 'Rk',right_on='Rk').merge(dt_passing,left_on = 'Rk',right_on='Rk').merge(dt_defensive,left_on = 'Rk',right_on='Rk')

    columns={"Player": "Player","Age": "Age","Squad": "Squad","MP": "PJ", "Min": "Min", "Gls_x": "Gls","Ast_x": "Ast","PK_x":"PK","CrdY":"TA","CrdR":"TR","xG_x":"xG","xAG_x":"xAG","PrgC":"PrgC","PrgP_x":"PrgP","PrgR":"PrgR", #Standar
         "Sh_x":"Sh","SoT":"SoT","G/Sh":"G/Sh","G/SoT":"G/SoT","Dist":"DistSh","FK":"FK",                                                                                   #Shooting
         "Cmp":"PasCompT","Cmp%.1":"PasCort","Cmp%.3":"PasLarg","1/3":"PasUltTer",                                                                                          #Passing
         "Tkl":"Tkl","Def 3rd":"TklDef","Mid 3rd":"TklMed","Att 3rd":"TklAtt","Int":"Int"
         } 

    claves = list(columns.values())

    df_mex_detalles.rename(
    columns=columns,
    inplace=True,
    )

    df_mex_detalles=df_mex_detalles[claves]
    df_mex_detalles.fillna(0,inplace=True)

    #######################################################################

    for index, row in df_mex_detalles.loc[df_mex_detalles.duplicated(subset=['Player','Age'])].iterrows():
        df_EquipoActual = df_mex_detalles.iloc[[index]]
        jugador = df_EquipoActual['Player'][index]
        age = df_EquipoActual['Age'][index]
        squad = df_EquipoActual['Squad'][index]
        df_EquipoAnterior = df_mex_detalles[(df_mex_detalles['Player'] == jugador) & 
                                (df_mex_detalles['Age'] == age) ]
                                #& 
                                #(df_mex_detalles['Squad'] != squad)]


        df_mex_detalles.loc[index, 'PJ'] = df_EquipoActual['PJ'][index] + df_EquipoAnterior['PJ'].values[0]
        df_mex_detalles.loc[index, 'Min'] = df_EquipoActual['Min'][index] + df_EquipoAnterior['Min'].values[0]
        df_mex_detalles.loc[index, 'Gls'] = df_EquipoActual['Gls'][index] + df_EquipoAnterior['Gls'].values[0]
        df_mex_detalles.loc[index, 'Ast'] = df_EquipoActual['Ast'][index] + df_EquipoAnterior['Ast'].values[0]
        df_mex_detalles.loc[index, 'PK'] = df_EquipoActual['PK'][index] + df_EquipoAnterior['PK'].values[0]
        df_mex_detalles.loc[index, 'TA'] = df_EquipoActual['TA'][index] + df_EquipoAnterior['TA'].values[0]
        df_mex_detalles.loc[index, 'TR'] = df_EquipoActual['TR'][index] + df_EquipoAnterior['TR'].values[0]
        df_mex_detalles.loc[index, 'xG'] = df_EquipoActual['xG'][index] + df_EquipoAnterior['xG'].values[0]
        df_mex_detalles.loc[index, 'xAG'] = df_EquipoActual['xAG'][index] + df_EquipoAnterior['xAG'].values[0]
        df_mex_detalles.loc[index, 'PrgC'] = df_EquipoActual['PrgC'][index] + df_EquipoAnterior['PrgC'].values[0]
        df_mex_detalles.loc[index, 'PrgP'] = df_EquipoActual['PrgP'][index] + df_EquipoAnterior['PrgP'].values[0]
        df_mex_detalles.loc[index, 'PrgR'] = df_EquipoActual['PrgR'][index] + df_EquipoAnterior['PrgR'].values[0]
        df_mex_detalles.loc[index, 'Sh'] = df_EquipoActual['Sh'][index] + df_EquipoAnterior['Sh'].values[0]
        df_mex_detalles.loc[index, 'SoT'] = df_EquipoActual['SoT'][index] + df_EquipoAnterior['SoT'].values[0]
        df_mex_detalles.loc[index, 'G/Sh'] = (df_EquipoActual['G/Sh'][index] + df_EquipoAnterior['G/Sh'].values[0])/2
        df_mex_detalles.loc[index, 'G/SoT'] = (df_EquipoActual['G/SoT'][index] + df_EquipoAnterior['G/SoT'].values[0])/2
        df_mex_detalles.loc[index, 'DistSh'] = (df_EquipoActual['DistSh'][index] + df_EquipoAnterior['DistSh'].values[0])/2
        df_mex_detalles.loc[index, 'FK'] = df_EquipoActual['FK'][index] + df_EquipoAnterior['FK'].values[0]
        df_mex_detalles.loc[index, 'PasCompT'] = df_EquipoActual['PasCompT'][index] + df_EquipoAnterior['PasCompT'].values[0]
        df_mex_detalles.loc[index, 'PasCort'] = df_EquipoActual['PasCort'][index] + df_EquipoAnterior['PasCort'].values[0]
        df_mex_detalles.loc[index, 'PasLarg'] = df_EquipoActual['PasLarg'][index] + df_EquipoAnterior['PasLarg'].values[0]
        df_mex_detalles.loc[index, 'PasUltTer'] = df_EquipoActual['PasUltTer'][index] + df_EquipoAnterior['PasUltTer'].values[0]
        df_mex_detalles.loc[index, 'Tkl'] = df_EquipoActual['Tkl'][index] + df_EquipoAnterior['Tkl'].values[0]
        df_mex_detalles.loc[index, 'TklDef'] = df_EquipoActual['TklDef'][index] + df_EquipoAnterior['TklDef'].values[0]
        df_mex_detalles.loc[index, 'TklMed'] = df_EquipoActual['TklMed'][index] + df_EquipoAnterior['TklMed'].values[0]
        df_mex_detalles.loc[index, 'TklAtt'] = df_EquipoActual['TklAtt'][index] + df_EquipoAnterior['TklAtt'].values[0]
        df_mex_detalles.loc[index, 'Int'] = df_EquipoActual['Int'][index] + df_EquipoAnterior['Int'].values[0]
    
    df_mex_detalles = df_mex_detalles.drop_duplicates(subset=['Player','Age'], keep="last")
    df_mex_detalles['Player']=df_mex_detalles[['Player','Squad']].apply(lambda fila: str(fila[0]) + " (" +  str(fila[1] + ") "), axis=1)

    df_mex_detalles = df_mex_detalles.drop(["Age", "Squad"], axis=1)

    dt=df_mex_detalles.copy()
    #######################################################################

    col1, col2= st.columns(2)

    with col1:

        expander = st.sidebar.expander("Selecionar Jugador")
        with expander:

            jugador_option= dt['Player'].unique().tolist()
            jugador_name=st.selectbox('Seleccionar',jugador_option,1)   
            
            
            # Pasamos las métricas a una matriz X y los nombres de los jugadores a una matriz y
            X, y = dt.iloc[:, 1:len(dt.columns)].values, dt.iloc[:, 0].values

            # Escalamos la matriz X
            X_std = StandardScaler().fit_transform(X)

            # Aplicamos PCA para obtener una matriz x pero con las dimensiones (X_pca)
            pca = PCA(n_components = len(dt.columns)-1)
            pca.fit(X_std)
            X_pca = pca.transform(X_std)    

            # Seleccionamos el número de componentes y mostramos (head) el dataframe con las dimensiones
            N_COMP = 10
            columns = []

            for col in range(1, N_COMP+1, 1):
                columns.append("PCA" + str(col))

            dt_pca_resultado = pd.DataFrame(data=X_pca[:,0:N_COMP], columns=columns, index = y)

            # Obtenemos la matriz de correlación (Pearson)
            corr_matrix = dt_pca_resultado.T.corr(method='pearson')

            NumPlayers = 10

            df_correlatedPlayers = GetSimilarPlayers(jugador_name, NumPlayers, corr_matrix)

 

        st.header('Jugadores similares a '+ str(jugador_name))

        df_prueba=df_correlatedPlayers.copy()  
        df_prueba['Correlation Factor']=df_prueba[['Correlation Factor']].apply(lambda fila: str(round((fila[0]*100),2)), axis=1)
        df_prueba.rename(
           columns={"PlayerName":"Jugador","Similar Player":"Jugador Similar","Correlation Factor":"% Similitud"},
            inplace=True,
            )
        df_prueba.set_index('Jugador', inplace=True)
        st.table(df_prueba)
       

    with col2:
         
        with expander:
            if len(df_correlatedPlayers)>0:

                player_option= df_correlatedPlayers['Similar Player'].unique().tolist()
                players_name=st.multiselect('Jugadore similares',player_option,player_option[0])
                metri=['PJ','Min','Gls','Ast','PK','TA','TR','xG','xAG','PrgC','PrgP','PrgR','Sh','SoT','G/Sh','G/SoT','DistSh','FK','PasCompT','PasCort','PasLarg','PasUltTer','Tkl','TklDef','TklMed','TklAtt','Int']           
                metri_s=['PJ','Gls','xG','Sh','PasCompT','Tkl','Ast','PrgP','PrgC']
                colums=st.multiselect('Seleccionar métricas',metri,metri_s)
                
                
                colums.insert(0,'Player')
                players_name.insert(0,jugador_name)
                players_name=sorted(players_name)              

                x = 0
                cadena_nombre=""
                while x < len(players_name):
                    cadena_nombre=cadena_nombre+"<"+players_name[x]+">"+" | "
                    x += 1

                df_Forward= dt[colums].copy()

                if len(players_name)==0:
                        mask= (df_Forward['Player']=='André-Pierre Gignac')|(df_Forward['Player']=='Alexis Vega')
                elif len(players_name)==1:
                        mask= (df_Forward['Player']==players_name[0])
                elif len(players_name)==2:
                        mask= (df_Forward['Player']==players_name[0])|(df_Forward['Player']==players_name[1])
                elif len(players_name)==3:
                        mask= (df_Forward['Player']==players_name[0])|(df_Forward['Player']==players_name[1])|(df_Forward['Player']==players_name[2])
                elif len(players_name)==4:
                        mask= (df_Forward['Player']==players_name[0])|(df_Forward['Player']==players_name[1])|(df_Forward['Player']==players_name[2])|(df_Forward['Player']==players_name[3])
                elif len(players_name)==5:
                        mask= (df_Forward['Player']==players_name[0])|(df_Forward['Player']==players_name[1])|(df_Forward['Player']==players_name[2])|(df_Forward['Player']==players_name[3])|(df_Forward['Player']==players_name[4])


                df_Forward=df_Forward[mask].reset_index()
                df_Forward=df_Forward.sort_values(by=['Player'])
                df_Forward=df_Forward.drop_duplicates(subset="Player",)

                if len(players_name)==0:
                    a_values=df_Forward.iloc[0].values.tolist()    
                    b_values=df_Forward.iloc[1].values.tolist()    
                    a_values=a_values[2:]
                    b_values=b_values[2:]

                elif len(players_name)==1:
                    a_values=df_Forward.iloc[0].values.tolist()      
                    a_values=a_values[2:]

                elif len(players_name)==2:
                    a_values=df_Forward.iloc[0].values.tolist()    
                    b_values=df_Forward.iloc[1].values.tolist()    
                    a_values=a_values[2:]
                    b_values=b_values[2:]

                elif len(players_name)==3:
                    a_values=df_Forward.iloc[0].values.tolist()    
                    b_values=df_Forward.iloc[1].values.tolist()    
                    c_values=df_Forward.iloc[2].values.tolist()    
                    a_values=a_values[2:]
                    b_values=b_values[2:]
                    c_values=c_values[2:]

                elif len(players_name)==4:
                    a_values=df_Forward.iloc[0].values.tolist()    
                    b_values=df_Forward.iloc[1].values.tolist()    
                    c_values=df_Forward.iloc[2].values.tolist()    
                    d_values=df_Forward.iloc[3].values.tolist()    
                    a_values=a_values[2:]
                    b_values=b_values[2:]
                    c_values=c_values[2:]   
                    d_values=d_values[2:]  

                elif len(players_name)==5:
                    a_values=df_Forward.iloc[0].values.tolist()    
                    b_values=df_Forward.iloc[1].values.tolist()    
                    c_values=df_Forward.iloc[2].values.tolist()    
                    d_values=df_Forward.iloc[3].values.tolist()    
                    e_values=df_Forward.iloc[4].values.tolist()   
                    a_values=a_values[2:]
                    b_values=b_values[2:]
                    c_values=c_values[2:]   
                    d_values=d_values[2:]  
                    e_values=e_values[2:]  

                params= list(df_Forward.columns)

                params= params[2:]
                #add ranges to list
                ranges=[]
                low = [] 
                high =[] 
                for x in params:
                    a=min(df_Forward[params][x])
                    
                    a= a - (a * .25)

                    b=max(df_Forward[params][x])
                    b=b-(b*.25)     

                    if a==b:
                        b=b+0000.1

                    low.append((a))
                    high.append((b))

                # Add anything to this list where having a lower number is better
                # this flips the statistic
                lower_is_better = ['Miscontrol']

                radar = Radar(params, low, high,
                        lower_is_better=lower_is_better,
                        # whether to round any of the labels to integers instead of decimal places
                        round_int=[False]*len(params),
                        num_rings=4,  # the number of concentric circles (excluding center circle)
                        # if the ring_width is more than the center_circle_radius then
                        # the center circle radius will be wider than the width of the concentric circles
                        ring_width=1, center_circle_radius=1)


                URL2 = ('https://raw.githubusercontent.com/googlefonts/SourceSerifProGFVersion/main/fonts/'
                        'SourceSerifPro-ExtraLight.ttf')
                serif_extra_light = FontManager(URL2)
                URL3 = ('https://raw.githubusercontent.com/google/fonts/main/ofl/rubikmonoone/'
                        'RubikMonoOne-Regular.ttf')
                rubik_regular = FontManager(URL3)
                URL4 = 'https://raw.githubusercontent.com/googlefonts/roboto/main/src/hinted/Roboto-Thin.ttf'
                robotto_thin = FontManager(URL4)
                URL5 = ('https://raw.githubusercontent.com/google/fonts/main/apache/robotoslab/'
                        'RobotoSlab%5Bwght%5D.ttf')
                robotto_bold = FontManager(URL5)

                # plot radar
                fig, ax = radar.setup_axis()
                rings_inner = radar.draw_circles(ax=ax, facecolor='#ffb2b2', edgecolor='#fc5f5f')

                if len(players_name)==0:      
                    radar1, vertices1 = radar.draw_radar_solid(a_values, ax=ax,
                                                            kwargs={'facecolor': '#aa65b2',
                                                                    'alpha': 0.6,
                                                                    'edgecolor': '#502a54',
                                                                    'lw': 3})

                    radar2, vertices2 = radar.draw_radar_solid(b_values, ax=ax,
                                                            kwargs={'facecolor': '#66d8ba',
                                                                    'alpha': 0.6,
                                                                    'edgecolor': '#216352',
                                                                    'lw': 3})

                    ax.scatter(vertices1[:, 0], vertices1[:, 1],
                            c='#aa65b2', edgecolors='#502a54', marker='o', s=150, zorder=2)
                    ax.scatter(vertices2[:, 0], vertices2[:, 1],
                            c='#66d8ba', edgecolors='#216352', marker='o', s=150, zorder=2)   

                elif len(players_name)==1:      
                    radar1, vertices1 = radar.draw_radar_solid(a_values, ax=ax,
                                                            kwargs={'facecolor': '#aa65b2',
                                                                    'alpha': 0.6,
                                                                    'edgecolor': '#216352',
                                                                    'lw': 3})

                    ax.scatter(vertices1[:, 0], vertices1[:, 1],
                            c='#aa65b2', edgecolors='#502a54', marker='o', s=150, zorder=2)

                elif len(players_name)==2:      
                    radar1, vertices1 = radar.draw_radar_solid(a_values, ax=ax,
                                                            kwargs={'facecolor': '#aa65b2',
                                                                    'alpha': 0.6,
                                                                    'edgecolor': '#502a54',
                                                                    'lw': 3})

                    radar2, vertices2 = radar.draw_radar_solid(b_values, ax=ax,
                                                            kwargs={'facecolor': '#66d8ba',
                                                                    'alpha': 0.6,
                                                                    'edgecolor': '#216352',
                                                                    'lw': 3})

                    ax.scatter(vertices1[:, 0], vertices1[:, 1],
                            c='#aa65b2', edgecolors='#502a54', marker='o', s=150, zorder=2)
                    ax.scatter(vertices2[:, 0], vertices2[:, 1],
                            c='#66d8ba', edgecolors='#216352', marker='o', s=150, zorder=2)
                
                elif len(players_name)==3:      
                    radar1, vertices1 = radar.draw_radar_solid(a_values, ax=ax,
                                                            kwargs={'facecolor': '#aa65b2',
                                                                    'alpha': 0.6,
                                                                    'edgecolor': '#216354',
                                                                    'lw': 3})

                    radar2, vertices2 = radar.draw_radar_solid(b_values, ax=ax,
                                                            kwargs={'facecolor': '#66d8ba',
                                                                    'alpha': 0.6,
                                                                    'edgecolor': '#216352',
                                                                    'lw': 3})

                    radar3, vertices3 = radar.draw_radar_solid(c_values, ax=ax,
                                                            kwargs={'facecolor': '#697cd4',
                                                                    'alpha': 0.6,
                                                                    'edgecolor': '#222b54',
                                                                    'lw': 3})     
                    ax.scatter(vertices1[:, 0], vertices1[:, 1],
                            c='#aa65b2', edgecolors='#502a54', marker='o', s=150, zorder=2)
                    ax.scatter(vertices2[:, 0], vertices2[:, 1],
                            c='#66d8ba', edgecolors='#216352', marker='o', s=150, zorder=2)
                    ax.scatter(vertices3[:, 0], vertices3[:, 1],
                            c='#697cd4', edgecolors='#222b54', marker='o', s=150, zorder=2)                                                                                             

                elif len(players_name)==4:      
                    radar1, vertices1 = radar.draw_radar_solid(a_values, ax=ax,
                                                            kwargs={'facecolor': '#aa65b2',
                                                                    'alpha': 0.6,
                                                                    'edgecolor': '#216354',
                                                                    'lw': 3})

                    radar2, vertices2 = radar.draw_radar_solid(b_values, ax=ax,
                                                            kwargs={'facecolor': '#66d8ba',
                                                                    'alpha': 0.6,
                                                                    'edgecolor': '#216352',
                                                                    'lw': 3})

                    radar3, vertices3 = radar.draw_radar_solid(c_values, ax=ax,
                                                            kwargs={'facecolor': '#697cd4',
                                                                    'alpha': 0.6,
                                                                    'edgecolor': '#222b54',
                                                                    'lw': 3})
                    radar4, vertices4 = radar.draw_radar_solid(d_values, ax=ax,
                                                            kwargs={'facecolor': '#778821',
                                                                    'alpha': 0.6,
                                                                    'edgecolor': '#4D5B05',
                                                                    'lw': 3})   
                                                                                                                            
                    ax.scatter(vertices1[:, 0], vertices1[:, 1],
                            c='#aa65b2', edgecolors='#502a54', marker='o', s=150, zorder=2)
                    ax.scatter(vertices2[:, 0], vertices2[:, 1],
                            c='#66d8ba', edgecolors='#216352', marker='o', s=150, zorder=2)
                    ax.scatter(vertices3[:, 0], vertices3[:, 1],
                            c='#697cd4', edgecolors='#222b54', marker='o', s=150, zorder=2)  
                    ax.scatter(vertices4[:, 0], vertices4[:, 1],
                            c='#778821', edgecolors='#4D5B05', marker='o', s=150, zorder=2)      

                elif len(players_name)==5:      
                    radar1, vertices1 = radar.draw_radar_solid(a_values, ax=ax,
                                                            kwargs={'facecolor': '#aa65b2',
                                                                    'alpha': 0.6,
                                                                    'edgecolor': '#216354',
                                                                    'lw': 3})

                    radar2, vertices2 = radar.draw_radar_solid(b_values, ax=ax,
                                                            kwargs={'facecolor': '#66d8ba',
                                                                    'alpha': 0.6,
                                                                    'edgecolor': '#216352',
                                                                    'lw': 3})

                    radar3, vertices3 = radar.draw_radar_solid(c_values, ax=ax,
                                                            kwargs={'facecolor': '#697cd4',
                                                                    'alpha': 0.6,
                                                                    'edgecolor': '#222b54',
                                                                    'lw': 3})
                    radar4, vertices4 = radar.draw_radar_solid(d_values, ax=ax,
                                                            kwargs={'facecolor': '#778821',
                                                                    'alpha': 0.6,
                                                                    'edgecolor': '#4D5B05',
                                                                    'lw': 3})   
                    radar5, vertices5 = radar.draw_radar_solid(e_values, ax=ax,
                                                            kwargs={'facecolor': '#a82228',
                                                                    'alpha': 0.6,
                                                                    'edgecolor': '#930e14',
                                                                    'lw': 3})   

                    ax.scatter(vertices1[:, 0], vertices1[:, 1],
                            c='#aa65b2', edgecolors='#502a54', marker='o', s=150, zorder=2)
                    ax.scatter(vertices2[:, 0], vertices2[:, 1],
                            c='#66d8ba', edgecolors='#216352', marker='o', s=150, zorder=2)
                    ax.scatter(vertices3[:, 0], vertices3[:, 1],
                            c='#697cd4', edgecolors='#222b54', marker='o', s=150, zorder=2)  
                    ax.scatter(vertices4[:, 0], vertices4[:, 1],
                            c='#778821', edgecolors='#4D5B05', marker='o', s=150, zorder=2)   
                    ax.scatter(vertices5[:, 0], vertices5[:, 1],
                            c='#a82228', edgecolors='#930e14', marker='o', s=150, zorder=2)   

                range_labels = radar.draw_range_labels(ax=ax, fontsize=20, fontproperties=robotto_bold.prop, color="#000000")
                param_labels = radar.draw_param_labels(ax=ax, fontsize=20, fontproperties=robotto_bold.prop, color="#000000")

                if len(players_name)==1:  
                    fig_text(
                        0.515, 0.99, cadena_nombre, size=22, fig=fig,
                        highlight_textprops=[{"color": '#aa65b2'}],                                                                
                        ha="center", fontproperties=robotto_bold.prop, color="#000000"
                    )
                elif len(players_name)==2:  
                            fig_text(
                        0.515, 0.99, cadena_nombre, size=22, fig=fig,
                        highlight_textprops=[{"color": '#aa65b2'}, {"color": '#66d8ba'}],                                                                
                        ha="center", fontproperties=robotto_bold.prop, color="#000000"
                    )
                elif len(players_name)==3:  
                            fig_text(
                        0.515, 0.99, cadena_nombre, size=22, fig=fig,
                        highlight_textprops=[{"color": '#aa65b2'}, {"color": '#66d8ba'}, {"color": '#697cd4'}],                                                                
                        ha="center", fontproperties=robotto_bold.prop, color="#000000"
                    )
                elif len(players_name)==4:  
                            fig_text(
                        0.515, 0.99, cadena_nombre, size=22, fig=fig,
                        highlight_textprops=[{"color": '#aa65b2'}, {"color": '#66d8ba'}, {"color": '#697cd4'}, {"color": '#778821'}],                                                                
                        ha="center", fontproperties=robotto_bold.prop, color="#000000"
                    )
                elif len(players_name)==5:  
                            fig_text(
                        0.515, 0.99, cadena_nombre, size=22, fig=fig,
                        highlight_textprops=[{"color": '#aa65b2'}, {"color": '#66d8ba'}, {"color": '#697cd4'}, {"color": '#778821'}, {"color": '#a82228'}],                                                                
                        ha="center", fontproperties=robotto_bold.prop, color="#000000"
                    )
            
                # add subtitle
                fig.text(
                    0.515, 0.942,
                    str(len(dt))+" Jugadores de la Liga MX (Primera División de México) | Temporada 2022-23",
                    size=20,
                    ha="center", fontproperties=robotto_bold.prop, color="#000000"
                )

                    # add credits
                CREDIT_1 = "data: statsbomb vía fbref"
                CREDIT_2 = "inspired by: @everdata9"

                fig.text(
                    0.99, 0.005, f"{CREDIT_1}\n{CREDIT_2}", size=12,
                    fontproperties=robotto_bold.prop, color="#000000",
                    ha="right"
                )


        st.write(fig)  
        
        

    #st.write('Para este proceso se utilizó el análisis de componentes principales o PCA, siendo esta una de las técnicas más clásicas de Machine Learning.') 
    #st.write('PCA (Principal Component Analysis) es uno de los algoritmos de reducción de dimensionalidad más populares, como hay muchas métricas diferentes para cada equipo y otras muchas métricas diferentes para cada jugador, lo que debemos hacer es trabajar con Dimensionality Reduction, en nuestro caso PCA, para comprimir los datasets y empezar a trabajar con las correlaciones. Así evitamos introducir ruido y problemas en nuestro sistema.')
    st.markdown('<div style="text-align: justify; border: 2px solid gray; padding: 10px; font-size: 20px;"><i>Para este proceso se utilizó el análisis de componentes principales o PCA, siendo esta una de las técnicas más clásicas de Machine Learning. PCA (Principal Component Analysis) es uno de los algoritmos de reducción de dimensionalidad más populares, como hay muchas métricas diferentes para cada equipo y otras muchas métricas diferentes para cada jugador, lo que debemos hacer es trabajar con Dimensionality Reduction, en nuestro caso PCA, para comprimir los datasets y empezar a trabajar con las correlaciones. Así evitamos introducir ruido y problemas en nuestro sistema.</i></div>', unsafe_allow_html=True)
    st.write('')
    st.write(fechaAct)
    st.header('MÉTRICAS')
    st.markdown(text_contents)
    st.header('POSICIONES')
    st.markdown(txt_Position)
elif selected == "Rendimiento Equipos":

    # Cargamos el dataset (en la carpeta data con nombre PCA)
    dt_standar = pd.read_csv('Data/mexico_standard_squad_22_23.csv', sep = ';')
    dt_shooting = pd.read_csv('Data/mexico_shooting_squad_22_23.csv', sep = ';')
    dt_passing = pd.read_csv('Data/mexico_passing_squad_22_23.csv', sep = ';')
    dt_defensive = pd.read_csv('Data/mexico_defensive_squad_22_23.csv', sep = ';')
    df_mex_detalles = pd.merge(dt_standar,dt_shooting,left_on = 'Squad',right_on='Squad').merge(dt_passing,left_on = 'Squad',right_on='Squad').merge(dt_defensive,left_on = 'Squad',right_on='Squad')

    columns={"Squad":"Squad","MP": "PJ", "Min": "Min", "Gls_x": "Gls","Ast_x": "Ast","PK_x":"PK","CrdY":"TA","CrdR":"TR","xG_x":"xG","xAG_x":"xAG","PrgC":"PrgC","PrgP_x":"PrgP", #Standar
         "Sh_x":"Sh","SoT":"SoT","G/Sh":"G/Sh","G/SoT":"G/SoT","Dist":"DistSh","FK":"FK",                                                                                   #Shooting
         "Cmp":"PasCompT","Cmp%.1":"PasCort","Cmp%.3":"PasLarg","1/3":"PasUltTer",                                                                                          #Passing
         "Tkl":"Tkl","Def 3rd":"TklDef","Mid 3rd":"TklMed","Att 3rd":"TklAtt","Int":"Int"
         } 

    claves = list(columns.values())

    df_mex_detalles.rename(
    columns=columns,
    inplace=True,
    )

    df_mex_detalles=df_mex_detalles[claves]
    df_mex_detalles.fillna(0,inplace=True)
    
    df= df_mex_detalles.copy()

    col1, col2 = st.columns(2)


    with col1:

        expander = st.sidebar.expander("Seleccionar Equipos")
        with expander:

            #squad_option= df['Squad'].unique().tolist()
            #position_name=st.selectbox('Seleccionar equipo',squad_option,4)        
          
            df_colums=df[['PJ','Min','Gls','Ast','PK','TA','TR','xG','xAG','PrgC','PrgP','Sh','SoT','G/Sh','G/SoT','DistSh','FK','PasCompT','PasCort','PasLarg','PasUltTer','Tkl','TklDef','TklMed','TklAtt','Int']]
            df_colums_s=df[['PJ','Gls','xG','Sh','PasCompT','Tkl','Ast','PrgP','PrgC']]
            player_option= df['Squad'].unique().tolist()
            metric_opction=df_colums.columns.tolist()
            metric_opction_s=df_colums_s.columns.tolist()
            players_name=st.multiselect('',player_option,['Atlas','América'])           

                   
            players_name=sorted(players_name)
            colums=st.multiselect('Seleccionar métricas',metric_opction,metric_opction_s)
            colums.insert(0,'Squad')
            #colums.insert(0,'Player')

            x = 0
            cadena_nombre=""
            while x < len(players_name):
                cadena_nombre=cadena_nombre+"<"+players_name[x]+">"+" | "
                x += 1

        df_Forward= df[colums].copy()
        if len(players_name)==0:
            mask= (df_Forward['Squad']=='Atlas')|(df_Forward['Squad']=='Guadalajara')
        elif len(players_name)==1:
            mask= (df_Forward['Squad']==players_name[0])
        elif len(players_name)==2:
            mask= (df_Forward['Squad']==players_name[0])|(df_Forward['Squad']==players_name[1])
        elif len(players_name)==3:
            mask= (df_Forward['Squad']==players_name[0])|(df_Forward['Squad']==players_name[1])|(df_Forward['Squad']==players_name[2])
        elif len(players_name)==4:
            mask= (df_Forward['Squad']==players_name[0])|(df_Forward['Squad']==players_name[1])|(df_Forward['Squad']==players_name[2])|(df_Forward['Squad']==players_name[3])
        elif len(players_name)==5:
            mask= (df_Forward['Squad']==players_name[0])|(df_Forward['Squad']==players_name[1])|(df_Forward['Squad']==players_name[2])|(df_Forward['Squad']==players_name[3])|(df_Forward['Squad']==players_name[4])


        df_Forward=df_Forward[mask].reset_index()
        df_Forward=df_Forward.sort_values(by=['Squad'])
        

        if len(players_name)==0:
            a_values=df_Forward.iloc[0].values.tolist()    
            b_values=df_Forward.iloc[1].values.tolist()    
            a_values=a_values[3:]
            b_values=b_values[3:]

        elif len(players_name)==1:
            a_values=df_Forward.iloc[0].values.tolist()      
            a_values=a_values[3:]

        elif len(players_name)==2:
            a_values=df_Forward.iloc[0].values.tolist()    
            b_values=df_Forward.iloc[1].values.tolist()    
            a_values=a_values[3:]
            b_values=b_values[3:]

        elif len(players_name)==3:
            a_values=df_Forward.iloc[0].values.tolist()    
            b_values=df_Forward.iloc[1].values.tolist()    
            c_values=df_Forward.iloc[2].values.tolist()    
            a_values=a_values[3:]
            b_values=b_values[3:]
            c_values=c_values[3:]

        elif len(players_name)==4:
            a_values=df_Forward.iloc[0].values.tolist()    
            b_values=df_Forward.iloc[1].values.tolist()    
            c_values=df_Forward.iloc[2].values.tolist()    
            d_values=df_Forward.iloc[3].values.tolist()    
            a_values=a_values[3:]
            b_values=b_values[3:]
            c_values=c_values[3:]   
            d_values=d_values[3:]  

        elif len(players_name)==5:
            a_values=df_Forward.iloc[0].values.tolist()    
            b_values=df_Forward.iloc[1].values.tolist()    
            c_values=df_Forward.iloc[2].values.tolist()    
            d_values=df_Forward.iloc[3].values.tolist()    
            e_values=df_Forward.iloc[4].values.tolist()   
            a_values=a_values[3:]
            b_values=b_values[3:]
            c_values=c_values[3:]   
            d_values=d_values[3:]  
            e_values=e_values[3:]  

        params= list(df_Forward.columns)

        params= params[3:]
        #add ranges to list
        ranges=[]
        low = [] 
        high =[] 
        for x in params:
            a=min(df_Forward[params][x])
            a= a - (a * .25)

            b=max(df_Forward[params][x])
            b=b-(b*.25)     

            if a==b:
                b=b+0000.1

            low.append((a))
            high.append((b))

        # Add anything to this list where having a lower number is better
        # this flips the statistic
        lower_is_better = ['Miscontrol']

        radar = Radar(params, low, high,
                lower_is_better=lower_is_better,
                # whether to round any of the labels to integers instead of decimal places
                round_int=[False]*len(params),
                num_rings=4,  # the number of concentric circles (excluding center circle)
                # if the ring_width is more than the center_circle_radius then
                # the center circle radius will be wider than the width of the concentric circles
                ring_width=1, center_circle_radius=1)


        URL2 = ('https://raw.githubusercontent.com/googlefonts/SourceSerifProGFVersion/main/fonts/'
                'SourceSerifPro-ExtraLight.ttf')
        serif_extra_light = FontManager(URL2)
        URL3 = ('https://raw.githubusercontent.com/google/fonts/main/ofl/rubikmonoone/'
                'RubikMonoOne-Regular.ttf')
        rubik_regular = FontManager(URL3)
        URL4 = 'https://raw.githubusercontent.com/googlefonts/roboto/main/src/hinted/Roboto-Thin.ttf'
        robotto_thin = FontManager(URL4)
        URL5 = ('https://raw.githubusercontent.com/google/fonts/main/apache/robotoslab/'
                'RobotoSlab%5Bwght%5D.ttf')
        robotto_bold = FontManager(URL5)

        # plot radar
        fig, ax = radar.setup_axis()
        rings_inner = radar.draw_circles(ax=ax, facecolor='#ffb2b2', edgecolor='#fc5f5f')

        if len(players_name)==0:      
            radar1, vertices1 = radar.draw_radar_solid(a_values, ax=ax,
                                                    kwargs={'facecolor': '#aa65b2',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#502a54',
                                                            'lw': 3})

            radar2, vertices2 = radar.draw_radar_solid(b_values, ax=ax,
                                                    kwargs={'facecolor': '#66d8ba',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#216352',
                                                            'lw': 3})

            ax.scatter(vertices1[:, 0], vertices1[:, 1],
                    c='#aa65b2', edgecolors='#502a54', marker='o', s=150, zorder=2)
            ax.scatter(vertices2[:, 0], vertices2[:, 1],
                    c='#66d8ba', edgecolors='#216352', marker='o', s=150, zorder=2)   

        elif len(players_name)==1:      
            radar1, vertices1 = radar.draw_radar_solid(a_values, ax=ax,
                                                    kwargs={'facecolor': '#aa65b2',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#216352',
                                                            'lw': 3})

            ax.scatter(vertices1[:, 0], vertices1[:, 1],
                    c='#aa65b2', edgecolors='#502a54', marker='o', s=150, zorder=2)

        elif len(players_name)==2:      
            radar1, vertices1 = radar.draw_radar_solid(a_values, ax=ax,
                                                    kwargs={'facecolor': '#aa65b2',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#502a54',
                                                            'lw': 3})

            radar2, vertices2 = radar.draw_radar_solid(b_values, ax=ax,
                                                    kwargs={'facecolor': '#66d8ba',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#216352',
                                                            'lw': 3})

            ax.scatter(vertices1[:, 0], vertices1[:, 1],
                    c='#aa65b2', edgecolors='#502a54', marker='o', s=150, zorder=2)
            ax.scatter(vertices2[:, 0], vertices2[:, 1],
                    c='#66d8ba', edgecolors='#216352', marker='o', s=150, zorder=2)
        
        elif len(players_name)==3:      
            radar1, vertices1 = radar.draw_radar_solid(a_values, ax=ax,
                                                    kwargs={'facecolor': '#aa65b2',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#216354',
                                                            'lw': 3})

            radar2, vertices2 = radar.draw_radar_solid(b_values, ax=ax,
                                                    kwargs={'facecolor': '#66d8ba',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#216352',
                                                            'lw': 3})

            radar3, vertices3 = radar.draw_radar_solid(c_values, ax=ax,
                                                    kwargs={'facecolor': '#697cd4',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#222b54',
                                                            'lw': 3})     
            ax.scatter(vertices1[:, 0], vertices1[:, 1],
                    c='#aa65b2', edgecolors='#502a54', marker='o', s=150, zorder=2)
            ax.scatter(vertices2[:, 0], vertices2[:, 1],
                    c='#66d8ba', edgecolors='#216352', marker='o', s=150, zorder=2)
            ax.scatter(vertices3[:, 0], vertices3[:, 1],
                    c='#697cd4', edgecolors='#222b54', marker='o', s=150, zorder=2)                                                                                             

        elif len(players_name)==4:      
            radar1, vertices1 = radar.draw_radar_solid(a_values, ax=ax,
                                                    kwargs={'facecolor': '#aa65b2',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#216354',
                                                            'lw': 3})

            radar2, vertices2 = radar.draw_radar_solid(b_values, ax=ax,
                                                    kwargs={'facecolor': '#66d8ba',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#216352',
                                                            'lw': 3})

            radar3, vertices3 = radar.draw_radar_solid(c_values, ax=ax,
                                                    kwargs={'facecolor': '#697cd4',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#222b54',
                                                            'lw': 3})
            radar4, vertices4 = radar.draw_radar_solid(d_values, ax=ax,
                                                    kwargs={'facecolor': '#778821',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#4D5B05',
                                                            'lw': 3})   
                                                                                                                    
            ax.scatter(vertices1[:, 0], vertices1[:, 1],
                    c='#aa65b2', edgecolors='#502a54', marker='o', s=150, zorder=2)
            ax.scatter(vertices2[:, 0], vertices2[:, 1],
                    c='#66d8ba', edgecolors='#216352', marker='o', s=150, zorder=2)
            ax.scatter(vertices3[:, 0], vertices3[:, 1],
                    c='#697cd4', edgecolors='#222b54', marker='o', s=150, zorder=2)  
            ax.scatter(vertices4[:, 0], vertices4[:, 1],
                    c='#778821', edgecolors='#4D5B05', marker='o', s=150, zorder=2)      

        elif len(players_name)==5:      
            radar1, vertices1 = radar.draw_radar_solid(a_values, ax=ax,
                                                    kwargs={'facecolor': '#aa65b2',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#216354',
                                                            'lw': 3})

            radar2, vertices2 = radar.draw_radar_solid(b_values, ax=ax,
                                                    kwargs={'facecolor': '#66d8ba',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#216352',
                                                            'lw': 3})

            radar3, vertices3 = radar.draw_radar_solid(c_values, ax=ax,
                                                    kwargs={'facecolor': '#697cd4',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#222b54',
                                                            'lw': 3})
            radar4, vertices4 = radar.draw_radar_solid(d_values, ax=ax,
                                                    kwargs={'facecolor': '#778821',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#4D5B05',
                                                            'lw': 3})   
            radar5, vertices5 = radar.draw_radar_solid(e_values, ax=ax,
                                                    kwargs={'facecolor': '#a82228',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#930e14',
                                                            'lw': 3})   

            ax.scatter(vertices1[:, 0], vertices1[:, 1],
                    c='#aa65b2', edgecolors='#502a54', marker='o', s=150, zorder=2)
            ax.scatter(vertices2[:, 0], vertices2[:, 1],
                    c='#66d8ba', edgecolors='#216352', marker='o', s=150, zorder=2)
            ax.scatter(vertices3[:, 0], vertices3[:, 1],
                    c='#697cd4', edgecolors='#222b54', marker='o', s=150, zorder=2)  
            ax.scatter(vertices4[:, 0], vertices4[:, 1],
                    c='#778821', edgecolors='#4D5B05', marker='o', s=150, zorder=2)   
            ax.scatter(vertices5[:, 0], vertices5[:, 1],
                    c='#a82228', edgecolors='#930e14', marker='o', s=150, zorder=2)   

        range_labels = radar.draw_range_labels(ax=ax, fontsize=20, fontproperties=robotto_bold.prop, color="#000000")
        param_labels = radar.draw_param_labels(ax=ax, fontsize=20, fontproperties=robotto_bold.prop, color="#000000")

        if len(players_name)==1:  
            fig_text(
                0.515, 0.99, cadena_nombre, size=22, fig=fig,
                highlight_textprops=[{"color": '#aa65b2'}],                                                                
                ha="center", fontproperties=robotto_bold.prop, color="#000000"
            )
        elif len(players_name)==2:  
                    fig_text(
                0.515, 0.99, cadena_nombre, size=22, fig=fig,
                highlight_textprops=[{"color": '#aa65b2'}, {"color": '#66d8ba'}],                                                                
                ha="center", fontproperties=robotto_bold.prop, color="#000000"
            )
        elif len(players_name)==3:  
                    fig_text(
                0.515, 0.99, cadena_nombre, size=22, fig=fig,
                highlight_textprops=[{"color": '#aa65b2'}, {"color": '#66d8ba'}, {"color": '#697cd4'}],                                                                
                ha="center", fontproperties=robotto_bold.prop, color="#000000"
            )
        elif len(players_name)==4:  
                    fig_text(
                0.515, 0.99, cadena_nombre, size=22, fig=fig,
                highlight_textprops=[{"color": '#aa65b2'}, {"color": '#66d8ba'}, {"color": '#697cd4'}, {"color": '#778821'}],                                                                
                ha="center", fontproperties=robotto_bold.prop, color="#000000"
            )
        elif len(players_name)==5:  
                    fig_text(
                0.515, 0.99, cadena_nombre, size=22, fig=fig,
                highlight_textprops=[{"color": '#aa65b2'}, {"color": '#66d8ba'}, {"color": '#697cd4'}, {"color": '#778821'}, {"color": '#a82228'}],                                                                
                ha="center", fontproperties=robotto_bold.prop, color="#000000"
            )
    
        # add subtitle
        fig.text(
            0.515, 0.942,
            "Liga MX (Primera División de México) | Temporada 2022-23",
            size=20,
            ha="center", fontproperties=robotto_bold.prop, color="#000000"
        )



            # add credits
        CREDIT_1 = "data: vía fbref"
        CREDIT_2 = "inspired by: @everdata9"

        fig.text(
            0.99, 0.005, f"{CREDIT_1}\n{CREDIT_2}", size=12,
            fontproperties=robotto_bold.prop, color="#000000",
            ha="right"
        )
        

        st.write(fig)        
    with col2:   

        st.header('****Tabla con datos comparativos****')       
        
        df1_transposed=df_Forward.iloc[:, 1:len(df_Forward.columns)]             
        df1_transposed.set_index('Squad', inplace=True)
        df1_transposed = df1_transposed.T 
        df1_transposed = add_trunc(df1_transposed)
        st.table(df1_transposed)
       # CSS to inject contained in a string
        hide_table_row_index = """
                    <style>
                        .css-165ax5l.edw49t13{
                        border: 1px solid;
                        }
                    </style>
                    """
    st.write(fechaAct)
    st.header('MÉTRICAS')
    st.markdown(text_contents)
    st.header('POSICIONES')
    st.markdown(txt_Position)
elif selected == "Equipos Similares":

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import plotly.graph_objects as go
     # Cargamos el dataset (en la carpeta data con nombre PCA)
    dt_standar = pd.read_csv('Data/mexico_standard_squad_22_23.csv', sep = ';')
    dt_shooting = pd.read_csv('Data/mexico_shooting_squad_22_23.csv', sep = ';')
    dt_passing = pd.read_csv('Data/mexico_passing_squad_22_23.csv', sep = ';')
    dt_defensive = pd.read_csv('Data/mexico_defensive_squad_22_23.csv', sep = ';')
    df_mex_detalles = pd.merge(dt_standar,dt_shooting,left_on = 'Squad',right_on='Squad').merge(dt_passing,left_on = 'Squad',right_on='Squad').merge(dt_defensive,left_on = 'Squad',right_on='Squad')

    columns={"Squad":"Squad","MP": "PJ", "Min": "Min", "Gls_x": "Gls","Ast_x": "Ast","PK_x":"PK","CrdY":"TA","CrdR":"TR","xG_x":"xG","xAG_x":"xAG","PrgC":"PrgC","PrgP_x":"PrgP", #Standar
         "Sh_x":"Sh","SoT":"SoT","G/Sh":"G/Sh","G/SoT":"G/SoT","Dist":"DistSh","FK":"FK",                                                                                   #Shooting
         "Cmp":"PasCompT","Cmp%.1":"PasCort","Cmp%.3":"PasLarg","1/3":"PasUltTer",                                                                                          #Passing
         "Tkl":"Tkl","Def 3rd":"TklDef","Mid 3rd":"TklMed","Att 3rd":"TklAtt","Int":"Int"
         } 

    claves = list(columns.values())

    df_mex_detalles.rename(
    columns=columns,
    inplace=True,
    )

    df_mex_detalles=df_mex_detalles[claves]

    df_mex_detalles.fillna(0,inplace=True)

    dt=df_mex_detalles.copy()
    #######################################################################

    col1, col2= st.columns(2)

    with col1:

        expander = st.sidebar.expander("Selecionar Equipo")
        with expander:

            jugador_option= dt['Squad'].unique().tolist()
            jugador_name=st.selectbox('Seleccionar',jugador_option,1)   
            
            
            # Pasamos las métricas a una matriz X y los nombres de los jugadores a una matriz y
            X, y = dt.iloc[:, 1:len(dt.columns)].values, dt.iloc[:, 0].values

            # Escalamos la matriz X
            X_std = StandardScaler().fit_transform(X)

            # Aplicamos PCA para obtener una matriz x pero con las dimensiones (X_pca)
            #pca = PCA(n_components = len(dt.columns)-1)
            pca = PCA(n_components = len(dt.columns)-9)
            pca.fit(X_std)
            X_pca = pca.transform(X_std)    

            # Seleccionamos el número de componentes y mostramos (head) el dataframe con las dimensiones
            N_COMP = 8
            columns = []

            for col in range(1, N_COMP+1, 1):
                columns.append("PCA" + str(col))

            dt_pca_resultado = pd.DataFrame(data=X_pca[:,0:N_COMP], columns=columns, index = y)

            # Obtenemos la matriz de correlación (Pearson)
            corr_matrix = dt_pca_resultado.T.corr(method='pearson')

            NumPlayers = 10

            df_correlatedPlayers = GetSimilarPlayers(jugador_name, NumPlayers, corr_matrix)

 

        st.header('Equipos similares a '+ str(jugador_name))

        df_prueba=df_correlatedPlayers.copy()  
        df_prueba['Correlation Factor']=df_prueba[['Correlation Factor']].apply(lambda fila: str(round((fila[0]*100),2)), axis=1)
        df_prueba.rename(
           columns={"PlayerName":"Equipo","Similar Player":"Equipos Similar","Correlation Factor":"% Similitud"},
            inplace=True,
            )
        df_prueba.set_index('Equipo', inplace=True)
        st.table(df_prueba)

    with col2:
         
        with expander:
            if len(df_correlatedPlayers)>0:

                player_option= df_correlatedPlayers['Similar Player'].unique().tolist()
                players_name=st.multiselect('Equipos similares',player_option,player_option[0])
                metri=['PJ','Min','Gls','Ast','PK','TA','TR','xG','xAG','PrgC','PrgP','Sh','SoT','G/Sh','G/SoT','DistSh','FK','PasCompT','PasCort','PasLarg','PasUltTer','Tkl','TklDef','TklMed','TklAtt','Int']           
                metri_s=['PJ','Gls','xG','Sh','PasCompT','Tkl','Ast','PrgP','PrgC']
                colums=st.multiselect('Seleccionar métricas',metri,metri_s)
                
                
                colums.insert(0,'Squad')
                players_name.insert(0,jugador_name)
                players_name=sorted(players_name)              
                
                x = 0
                cadena_nombre=""
                while x < len(players_name):
                    cadena_nombre=cadena_nombre+"<"+players_name[x]+">"+" | "
                    x += 1

                df_Forward= dt[colums].copy()

                if len(players_name)==0:
                        mask= (df_Forward['Squad']=='América')|(df_Forward['Player']=='Atlas')
                elif len(players_name)==1:
                        mask= (df_Forward['Squad']==players_name[0])
                elif len(players_name)==2:
                        mask= (df_Forward['Squad']==players_name[0])|(df_Forward['Squad']==players_name[1])
                elif len(players_name)==3:
                        mask= (df_Forward['Squad']==players_name[0])|(df_Forward['Squad']==players_name[1])|(df_Forward['Squad']==players_name[2])
                elif len(players_name)==4:
                        mask= (df_Forward['Squad']==players_name[0])|(df_Forward['Squad']==players_name[1])|(df_Forward['Squad']==players_name[2])|(df_Forward['Squad']==players_name[3])
                elif len(players_name)==5:
                        mask= (df_Forward['Player']==players_name[0])|(df_Forward['Squad']==players_name[1])|(df_Forward['Squad']==players_name[2])|(df_Forward['Squad']==players_name[3])|(df_Forward['Squad']==players_name[4])


                df_Forward=df_Forward[mask].reset_index()
                df_Forward=df_Forward.sort_values(by=['Squad'])
               

                if len(players_name)==0:
                    a_values=df_Forward.iloc[0].values.tolist()    
                    b_values=df_Forward.iloc[1].values.tolist()    
                    a_values=a_values[2:]
                    b_values=b_values[2:]

                elif len(players_name)==1:
                    a_values=df_Forward.iloc[0].values.tolist()      
                    a_values=a_values[2:]

                elif len(players_name)==2:
                    a_values=df_Forward.iloc[0].values.tolist()    
                    b_values=df_Forward.iloc[1].values.tolist()    
                    a_values=a_values[2:]
                    b_values=b_values[2:]

                elif len(players_name)==3:
                    a_values=df_Forward.iloc[0].values.tolist()    
                    b_values=df_Forward.iloc[1].values.tolist()    
                    c_values=df_Forward.iloc[2].values.tolist()    
                    a_values=a_values[2:]
                    b_values=b_values[2:]
                    c_values=c_values[2:]
                    

                elif len(players_name)==4:
                    a_values=df_Forward.iloc[0].values.tolist()    
                    b_values=df_Forward.iloc[1].values.tolist()    
                    c_values=df_Forward.iloc[2].values.tolist()    
                    d_values=df_Forward.iloc[3].values.tolist()    
                    a_values=a_values[2:]
                    b_values=b_values[2:]
                    c_values=c_values[2:]   
                    d_values=d_values[2:]  

                elif len(players_name)==5:
                    a_values=df_Forward.iloc[0].values.tolist()    
                    b_values=df_Forward.iloc[1].values.tolist()    
                    c_values=df_Forward.iloc[2].values.tolist()    
                    d_values=df_Forward.iloc[3].values.tolist()    
                    e_values=df_Forward.iloc[4].values.tolist()   
                    a_values=a_values[2:]
                    b_values=b_values[2:]
                    c_values=c_values[2:]   
                    d_values=d_values[2:]  
                    e_values=e_values[2:]  

                params= list(df_Forward.columns)
                params= params[2:]
                #add ranges to list
                ranges=[]
                low = [] 
                high =[] 
                for x in params:
                    a=min(df_Forward[params][x])
                    
                    a= a - (a * .25)

                    b=max(df_Forward[params][x])
                    b=b-(b*.25)     

                    if a==b:
                        b=b+0000.1

                    low.append((a))
                    high.append((b))

                # Add anything to this list where having a lower number is better
                # this flips the statistic
                lower_is_better = ['Miscontrol']

                radar = Radar(params, low, high,
                        lower_is_better=lower_is_better,
                        # whether to round any of the labels to integers instead of decimal places
                        round_int=[False]*len(params),
                        num_rings=4,  # the number of concentric circles (excluding center circle)
                        # if the ring_width is more than the center_circle_radius then
                        # the center circle radius will be wider than the width of the concentric circles
                        ring_width=1, center_circle_radius=1)


                URL2 = ('https://raw.githubusercontent.com/googlefonts/SourceSerifProGFVersion/main/fonts/'
                        'SourceSerifPro-ExtraLight.ttf')
                serif_extra_light = FontManager(URL2)
                URL3 = ('https://raw.githubusercontent.com/google/fonts/main/ofl/rubikmonoone/'
                        'RubikMonoOne-Regular.ttf')
                rubik_regular = FontManager(URL3)
                URL4 = 'https://raw.githubusercontent.com/googlefonts/roboto/main/src/hinted/Roboto-Thin.ttf'
                robotto_thin = FontManager(URL4)
                URL5 = ('https://raw.githubusercontent.com/google/fonts/main/apache/robotoslab/'
                        'RobotoSlab%5Bwght%5D.ttf')
                robotto_bold = FontManager(URL5)

                # plot radar
                fig, ax = radar.setup_axis()
                rings_inner = radar.draw_circles(ax=ax, facecolor='#ffb2b2', edgecolor='#fc5f5f')

                if len(players_name)==0:      
                    radar1, vertices1 = radar.draw_radar_solid(a_values, ax=ax,
                                                            kwargs={'facecolor': '#aa65b2',
                                                                    'alpha': 0.6,
                                                                    'edgecolor': '#502a54',
                                                                    'lw': 3})

                    radar2, vertices2 = radar.draw_radar_solid(b_values, ax=ax,
                                                            kwargs={'facecolor': '#66d8ba',
                                                                    'alpha': 0.6,
                                                                    'edgecolor': '#216352',
                                                                    'lw': 3})

                    ax.scatter(vertices1[:, 0], vertices1[:, 1],
                            c='#aa65b2', edgecolors='#502a54', marker='o', s=150, zorder=2)
                    ax.scatter(vertices2[:, 0], vertices2[:, 1],
                            c='#66d8ba', edgecolors='#216352', marker='o', s=150, zorder=2)   

                elif len(players_name)==1:      
                    radar1, vertices1 = radar.draw_radar_solid(a_values, ax=ax,
                                                            kwargs={'facecolor': '#aa65b2',
                                                                    'alpha': 0.6,
                                                                    'edgecolor': '#216352',
                                                                    'lw': 3})

                    ax.scatter(vertices1[:, 0], vertices1[:, 1],
                            c='#aa65b2', edgecolors='#502a54', marker='o', s=150, zorder=2)

                elif len(players_name)==2:      
                    radar1, vertices1 = radar.draw_radar_solid(a_values, ax=ax,
                                                            kwargs={'facecolor': '#aa65b2',
                                                                    'alpha': 0.6,
                                                                    'edgecolor': '#502a54',
                                                                    'lw': 3})

                    radar2, vertices2 = radar.draw_radar_solid(b_values, ax=ax,
                                                            kwargs={'facecolor': '#66d8ba',
                                                                    'alpha': 0.6,
                                                                    'edgecolor': '#216352',
                                                                    'lw': 3})

                    ax.scatter(vertices1[:, 0], vertices1[:, 1],
                            c='#aa65b2', edgecolors='#502a54', marker='o', s=150, zorder=2)
                    ax.scatter(vertices2[:, 0], vertices2[:, 1],
                            c='#66d8ba', edgecolors='#216352', marker='o', s=150, zorder=2)
                
                elif len(players_name)==3:      
                    radar1, vertices1 = radar.draw_radar_solid(a_values, ax=ax,
                                                            kwargs={'facecolor': '#aa65b2',
                                                                    'alpha': 0.6,
                                                                    'edgecolor': '#216354',
                                                                    'lw': 3})

                    radar2, vertices2 = radar.draw_radar_solid(b_values, ax=ax,
                                                            kwargs={'facecolor': '#66d8ba',
                                                                    'alpha': 0.6,
                                                                    'edgecolor': '#216352',
                                                                    'lw': 3})

                    radar3, vertices3 = radar.draw_radar_solid(c_values, ax=ax,
                                                            kwargs={'facecolor': '#697cd4',
                                                                    'alpha': 0.6,
                                                                    'edgecolor': '#222b54',
                                                                    'lw': 3})     
                    ax.scatter(vertices1[:, 0], vertices1[:, 1],
                            c='#aa65b2', edgecolors='#502a54', marker='o', s=150, zorder=2)
                    ax.scatter(vertices2[:, 0], vertices2[:, 1],
                            c='#66d8ba', edgecolors='#216352', marker='o', s=150, zorder=2)
                    ax.scatter(vertices3[:, 0], vertices3[:, 1],
                            c='#697cd4', edgecolors='#222b54', marker='o', s=150, zorder=2)                                                                                             

                elif len(players_name)==4:      
                    radar1, vertices1 = radar.draw_radar_solid(a_values, ax=ax,
                                                            kwargs={'facecolor': '#aa65b2',
                                                                    'alpha': 0.6,
                                                                    'edgecolor': '#216354',
                                                                    'lw': 3})

                    radar2, vertices2 = radar.draw_radar_solid(b_values, ax=ax,
                                                            kwargs={'facecolor': '#66d8ba',
                                                                    'alpha': 0.6,
                                                                    'edgecolor': '#216352',
                                                                    'lw': 3})

                    radar3, vertices3 = radar.draw_radar_solid(c_values, ax=ax,
                                                            kwargs={'facecolor': '#697cd4',
                                                                    'alpha': 0.6,
                                                                    'edgecolor': '#222b54',
                                                                    'lw': 3})
                    radar4, vertices4 = radar.draw_radar_solid(d_values, ax=ax,
                                                            kwargs={'facecolor': '#778821',
                                                                    'alpha': 0.6,
                                                                    'edgecolor': '#4D5B05',
                                                                    'lw': 3})   
                                                                                                                            
                    ax.scatter(vertices1[:, 0], vertices1[:, 1],
                            c='#aa65b2', edgecolors='#502a54', marker='o', s=150, zorder=2)
                    ax.scatter(vertices2[:, 0], vertices2[:, 1],
                            c='#66d8ba', edgecolors='#216352', marker='o', s=150, zorder=2)
                    ax.scatter(vertices3[:, 0], vertices3[:, 1],
                            c='#697cd4', edgecolors='#222b54', marker='o', s=150, zorder=2)  
                    ax.scatter(vertices4[:, 0], vertices4[:, 1],
                            c='#778821', edgecolors='#4D5B05', marker='o', s=150, zorder=2)      

                elif len(players_name)==5:      
                    radar1, vertices1 = radar.draw_radar_solid(a_values, ax=ax,
                                                            kwargs={'facecolor': '#aa65b2',
                                                                    'alpha': 0.6,
                                                                    'edgecolor': '#216354',
                                                                    'lw': 3})

                    radar2, vertices2 = radar.draw_radar_solid(b_values, ax=ax,
                                                            kwargs={'facecolor': '#66d8ba',
                                                                    'alpha': 0.6,
                                                                    'edgecolor': '#216352',
                                                                    'lw': 3})

                    radar3, vertices3 = radar.draw_radar_solid(c_values, ax=ax,
                                                            kwargs={'facecolor': '#697cd4',
                                                                    'alpha': 0.6,
                                                                    'edgecolor': '#222b54',
                                                                    'lw': 3})
                    radar4, vertices4 = radar.draw_radar_solid(d_values, ax=ax,
                                                            kwargs={'facecolor': '#778821',
                                                                    'alpha': 0.6,
                                                                    'edgecolor': '#4D5B05',
                                                                    'lw': 3})   
                    radar5, vertices5 = radar.draw_radar_solid(e_values, ax=ax,
                                                            kwargs={'facecolor': '#a82228',
                                                                    'alpha': 0.6,
                                                                    'edgecolor': '#930e14',
                                                                    'lw': 3})   

                    ax.scatter(vertices1[:, 0], vertices1[:, 1],
                            c='#aa65b2', edgecolors='#502a54', marker='o', s=150, zorder=2)
                    ax.scatter(vertices2[:, 0], vertices2[:, 1],
                            c='#66d8ba', edgecolors='#216352', marker='o', s=150, zorder=2)
                    ax.scatter(vertices3[:, 0], vertices3[:, 1],
                            c='#697cd4', edgecolors='#222b54', marker='o', s=150, zorder=2)  
                    ax.scatter(vertices4[:, 0], vertices4[:, 1],
                            c='#778821', edgecolors='#4D5B05', marker='o', s=150, zorder=2)   
                    ax.scatter(vertices5[:, 0], vertices5[:, 1],
                            c='#a82228', edgecolors='#930e14', marker='o', s=150, zorder=2)   

                range_labels = radar.draw_range_labels(ax=ax, fontsize=20, fontproperties=robotto_bold.prop, color="#000000")
                param_labels = radar.draw_param_labels(ax=ax, fontsize=20, fontproperties=robotto_bold.prop, color="#000000")

                if len(players_name)==1:  
                    fig_text(
                        0.515, 0.99, cadena_nombre, size=22, fig=fig,
                        highlight_textprops=[{"color": '#aa65b2'}],                                                                
                        ha="center", fontproperties=robotto_bold.prop, color="#000000"
                    )
                elif len(players_name)==2:  
                            fig_text(
                        0.515, 0.99, cadena_nombre, size=22, fig=fig,
                        highlight_textprops=[{"color": '#aa65b2'}, {"color": '#66d8ba'}],                                                                
                        ha="center", fontproperties=robotto_bold.prop, color="#000000"
                    )
                elif len(players_name)==3:  
                            fig_text(
                        0.515, 0.99, cadena_nombre, size=22, fig=fig,
                        highlight_textprops=[{"color": '#aa65b2'}, {"color": '#66d8ba'}, {"color": '#697cd4'}],                                                                
                        ha="center", fontproperties=robotto_bold.prop, color="#000000"
                    )
                elif len(players_name)==4:  
                            fig_text(
                        0.515, 0.99, cadena_nombre, size=22, fig=fig,
                        highlight_textprops=[{"color": '#aa65b2'}, {"color": '#66d8ba'}, {"color": '#697cd4'}, {"color": '#778821'}],                                                                
                        ha="center", fontproperties=robotto_bold.prop, color="#000000"
                    )
                elif len(players_name)==5:  
                            fig_text(
                        0.515, 0.99, cadena_nombre, size=22, fig=fig,
                        highlight_textprops=[{"color": '#aa65b2'}, {"color": '#66d8ba'}, {"color": '#697cd4'}, {"color": '#778821'}, {"color": '#a82228'}],                                                                
                        ha="center", fontproperties=robotto_bold.prop, color="#000000"
                    )
            
                # add subtitle
                fig.text(
                    0.515, 0.942,
                    str(len(dt))+" Equipos de la Liga MX (Primera División de México) | Temporada 2022-23",
                    size=20,
                    ha="center", fontproperties=robotto_bold.prop, color="#000000"
                )

                    # add credits
                CREDIT_1 = "data: statsbomb vía fbref"
                CREDIT_2 = "inspired by: @everdata9"

                fig.text(
                    0.99, 0.005, f"{CREDIT_1}\n{CREDIT_2}", size=12,
                    fontproperties=robotto_bold.prop, color="#000000",
                    ha="right"
                )


        st.write(fig)   
        

    #st.write('Para este proceso se utilizó el análisis de componentes principales o PCA, siendo esta una de las técnicas más clásicas de Machine Learning.') 
    #st.write('PCA (Principal Component Analysis) es uno de los algoritmos de reducción de dimensionalidad más populares, como hay muchas métricas diferentes para cada equipo y otras muchas métricas diferentes para cada jugador, lo que debemos hacer es trabajar con Dimensionality Reduction, en nuestro caso PCA, para comprimir los datasets y empezar a trabajar con las correlaciones. Así evitamos introducir ruido y problemas en nuestro sistema.')

    st.markdown('<div style="text-align: justify; border: 2px solid gray; padding: 10px; font-size: 20px;"><i>Para este proceso se utilizó el análisis de componentes principales o PCA, siendo esta una de las técnicas más clásicas de Machine Learning. PCA (Principal Component Analysis) es uno de los algoritmos de reducción de dimensionalidad más populares, como hay muchas métricas diferentes para cada equipo y otras muchas métricas diferentes para cada jugador, lo que debemos hacer es trabajar con Dimensionality Reduction, en nuestro caso PCA, para comprimir los datasets y empezar a trabajar con las correlaciones. Así evitamos introducir ruido y problemas en nuestro sistema.</i></div>', unsafe_allow_html=True)
    st.write('')
    st.write(fechaAct)
    st.header('MÉTRICAS')
    st.markdown(text_contents)
    st.header('POSICIONES')
    st.markdown(txt_Position)
elif selected == "Los Ángeles FC":


      # Cargamos el dataset (en la carpeta data con nombre PCA)
    dt_standar = pd.read_csv('Data/lafc_standar_23.csv', sep = ';')
    dt_shooting = pd.read_csv('Data/lafc_shooting_23.csv', sep = ';')
    dt_passing = pd.read_csv('Data/lafc_passing_23.csv', sep = ';')
    dt_defensive = pd.read_csv('Data/lafc_defensive_23.csv', sep = ';')

    df_mex_detalles = pd.merge(dt_standar,dt_shooting,left_on = 'Age',right_on='Age').merge(dt_passing,left_on = 'Age',right_on='Age').merge(dt_defensive,left_on = 'Age',right_on='Age')

    columns={"Player": "Player","Pos":"Pos","Age": "Age","MP": "PJ", "Min": "Min", "Gls_x": "Gls","Ast_x": "Ast","PK_x":"PK","CrdY":"TA","CrdR":"TR","xG_x":"xG","xAG_x":"xAG","PrgC":"PrgC","PrgP_x":"PrgP","PrgR":"PrgR", #Standar
         "Sh_x":"Sh","SoT":"SoT","G/Sh":"G/Sh","G/SoT":"G/SoT","Dist":"DistSh","FK":"FK",                                                                                   #Shooting
         "Cmp":"PasCompT","Cmp%.1":"PasCort","Cmp%.3":"PasLarg","1/3":"PasUltTer",                                                                                          #Passing
         "Tkl":"Tkl","Def 3rd":"TklDef","Mid 3rd":"TklMed","Att 3rd":"TklAtt","Int":"Int"
         } 

    claves = list(columns.values())

    df_mex_detalles.rename(
    columns=columns,
    inplace=True,
    )

    df_mex_detalles=df_mex_detalles[claves]
    df_mex_detalles.fillna(0,inplace=True)

    df= df_mex_detalles.copy()

    col1, col2 = st.columns(2)


    with col1:

        expander = st.sidebar.expander("Seleccionar Jugadores")
        with expander:
            position_option= df['Pos'].unique().tolist()
            position_name=st.selectbox('Seleccionar posición',position_option,1)        
            if position_name=='FW':
                df_colums=df[['PJ','Min','Gls','Ast','PK','TA','TR','xG','xAG','PrgC','PrgP','PrgR','Sh','SoT','G/Sh','G/SoT','DistSh','FK','PasCompT','PasCort','PasLarg','PasUltTer','Tkl','TklDef','TklMed','TklAtt','Int']]
                df_colums_s=df[['PJ','Gls','xG','Sh','PasCompT','Tkl','Ast','PrgP','PrgC']]
                player_option= df[(df['Pos']=='FW')]['Player'].unique().tolist()
                metric_opction=df_colums.columns.tolist()
                metric_opction_s=df_colums_s.columns.tolist()
                players_name=st.multiselect('Seleccionar jugador',player_option,['Denis Bouanga','Kwadwo Opoku'])             

            elif position_name=='DF':
                df_colums=df[['PJ','Min','Gls','Ast','PK','TA','TR','xG','xAG','PrgC','PrgP','PrgR','Sh','SoT','G/Sh','G/SoT','DistSh','FK','PasCompT','PasCort','PasLarg','PasUltTer','Tkl','TklDef','TklMed','TklAtt','Int']]
                df_colums_s=df[['PJ','Gls','xG','Sh','PasCompT','Tkl','Ast','PrgP','PrgC']]
                player_option= df[(df['Pos']=='DF')]['Player'].unique().tolist()
                metric_opction=df_colums.columns.tolist()
                metric_opction_s=df_colums_s.columns.tolist()
                players_name=st.multiselect('Seleccionar jugador',player_option,['Ryan Hollingshead','Giorgio Chiellini'])

            elif position_name=='MF':
                df_colums=df[['PJ','Min','Gls','Ast','PK','TA','TR','xG','xAG','PrgC','PrgP','PrgR','Sh','SoT','G/Sh','G/SoT','DistSh','FK','PasCompT','PasCort','PasLarg','PasUltTer','Tkl','TklDef','TklMed','TklAtt','Int']]
                df_colums_s=df[['PJ','Gls','xG','Sh','PasCompT','Tkl','Ast','PrgP','PrgC']]
                player_option= df[(df['Pos']=='MF')]['Player'].unique().tolist()
                metric_opction=df_colums.columns.tolist()
                metric_opction_s=df_colums_s.columns.tolist()
                players_name=st.multiselect('Seleccionar jugador',player_option,['Ilie Sánchez','Kellyn Acosta'])

            elif position_name=='MFDF':
                df_colums=df[['PJ','Min','Gls','Ast','PK','TA','TR','xG','xAG','PrgC','PrgP','PrgR','Sh','SoT','G/Sh','G/SoT','DistSh','FK','PasCompT','PasCort','PasLarg','PasUltTer','Tkl','TklDef','TklMed','TklAtt','Int']]
                df_colums_s=df[['PJ','Gls','xG','Sh','PasCompT','Tkl','Ast','PrgP','PrgC']]
                player_option= df[(df['Pos']=='MFDF')]['Player'].unique().tolist()
                metric_opction=df_colums.columns.tolist()
                metric_opction_s=df_colums_s.columns.tolist()
                players_name=st.multiselect('Seleccionar jugador',player_option,[])

            elif position_name=='FWMF':
                df_colums=df[['PJ','Min','Gls','Ast','PK','TA','TR','xG','xAG','PrgC','PrgP','PrgR','Sh','SoT','G/Sh','G/SoT','DistSh','FK','PasCompT','PasCort','PasLarg','PasUltTer','Tkl','TklDef','TklMed','TklAtt','Int']]
                df_colums_s=df[['PJ','Gls','xG','Sh','PasCompT','Tkl','Ast','PrgP','PrgC']]
                player_option= df[(df['Pos']=='FWMF')]['Player'].unique().tolist()
                metric_opction=df_colums.columns.tolist()
                metric_opction_s=df_colums_s.columns.tolist()
                players_name=st.multiselect('Seleccionar jugador',player_option,[])
             
            elif position_name=='DFMF':
                df_colums=df[['PJ','Min','Gls','Ast','PK','TA','TR','xG','xAG','PrgC','PrgP','PrgR','Sh','SoT','G/Sh','G/SoT','DistSh','FK','PasCompT','PasCort','PasLarg','PasUltTer','Tkl','TklDef','TklMed','TklAtt','Int']]
                df_colums_s=df[['PJ','Gls','xG','Sh','PasCompT','Tkl','Ast','PrgP','PrgC']]
                player_option= df[(df['Pos']=='DFMF')]['Player'].unique().tolist()
                metric_opction=df_colums.columns.tolist()
                metric_opction_s=df_colums_s.columns.tolist()
                players_name=st.multiselect('Seleccionar jugador',player_option,['Sergi Palencia'])
            
            elif position_name=='DFFW':
                df_colums=df[['PJ','Min','Gls','Ast','PK','TA','TR','xG','xAG','PrgC','PrgP','PrgR','Sh','SoT','G/Sh','G/SoT','DistSh','FK','PasCompT','PasCort','PasLarg','PasUltTer','Tkl','TklDef','TklMed','TklAtt','Int']]
                df_colums_s=df[['PJ','Gls','xG','Sh','PasCompT','Tkl','Ast','PrgP','PrgC']]
                player_option= df[(df['Pos']=='DFFW')]['Player'].unique().tolist()
                metric_opction=df_colums.columns.tolist()
                metric_opction_s=df_colums_s.columns.tolist()
                players_name=st.multiselect('Seleccionar jugador',player_option,[])

            elif position_name=='MFFW':
                df_colums=df[['PJ','Min','Gls','Ast','PK','TA','TR','xG','xAG','PrgC','PrgP','PrgR','Sh','SoT','G/Sh','G/SoT','DistSh','FK','PasCompT','PasCort','PasLarg','PasUltTer','Tkl','TklDef','TklMed','TklAtt','Int']]
                df_colums_s=df[['PJ','Gls','xG','Sh','PasCompT','Tkl','Ast','PrgP','PrgC']]
                player_option= df[(df['Pos']=='MFFW')]['Player'].unique().tolist()
                metric_opction=df_colums.columns.tolist()
                metric_opction_s=df_colums_s.columns.tolist()
                players_name=st.multiselect('Seleccionar jugador',player_option,[])           

            elif position_name=='FWDF':
                df_colums=df[['PJ','Min','Gls','Ast','PK','TA','TR','xG','xAG','PrgC','PrgP','PrgR','Sh','SoT','G/Sh','G/SoT','DistSh','FK','PasCompT','PasCort','PasLarg','PasUltTer','Tkl','TklDef','TklMed','TklAtt','Int']]
                df_colums_s=df[['PJ','Gls','xG','Sh','PasCompT','Tkl','Ast','PrgP','PrgC']]
                player_option= df[(df['Pos']=='FWDF')]['Player'].unique().tolist()
                metric_opction=df_colums.columns.tolist()
                metric_opction_s=df_colums_s.columns.tolist()
                players_name=st.multiselect('Seleccionar jugador',player_option,[])        

            elif position_name=='GK':
                df_colums=df[['PJ','Min','Gls','Ast','PK','TA','TR','xG','xAG','PrgC','PrgP','PrgR','Sh','SoT','G/Sh','G/SoT','DistSh','FK','PasCompT','PasCort','PasLarg','PasUltTer','Tkl','TklDef','TklMed','TklAtt','Int']]
                df_colums_s=df[['PJ','Gls','xG','Sh','PasCompT','Tkl','Ast','PrgP','PrgC']]
                player_option= df[(df['Pos']=='GK')]['Player'].unique().tolist()
                metric_opction=df_colums.columns.tolist()
                metric_opction_s=df_colums_s.columns.tolist()
                players_name=st.multiselect('Seleccionar jugador',player_option,['John McCarthy'])   

            players_name=sorted(players_name)
            colums=st.multiselect('Seleccionar métricas',metric_opction,metric_opction_s)
            #colums.insert(0,'Squad')
            colums.insert(0,'Player')

            x = 0
            cadena_nombre=""
            while x < len(players_name):
                cadena_nombre=cadena_nombre+"<"+players_name[x]+">"+" | "
                x += 1

        df_Forward= df[colums].copy()

        if len(players_name)==0:
            mask= (df_Forward['Player']=='Julián Quiñones (Atlas)')|(df_Forward['Player']=='Alexis Vega (Guadalajara)')
        elif len(players_name)==1:
            mask= (df_Forward['Player']==players_name[0])
        elif len(players_name)==2:
            mask= (df_Forward['Player']==players_name[0])|(df_Forward['Player']==players_name[1])
        elif len(players_name)==3:
            mask= (df_Forward['Player']==players_name[0])|(df_Forward['Player']==players_name[1])|(df_Forward['Player']==players_name[2])
        elif len(players_name)==4:
            mask= (df_Forward['Player']==players_name[0])|(df_Forward['Player']==players_name[1])|(df_Forward['Player']==players_name[2])|(df_Forward['Player']==players_name[3])
        elif len(players_name)==5:
            mask= (df_Forward['Player']==players_name[0])|(df_Forward['Player']==players_name[1])|(df_Forward['Player']==players_name[2])|(df_Forward['Player']==players_name[3])|(df_Forward['Player']==players_name[4])


        df_Forward=df_Forward[mask].reset_index()
        df_Forward=df_Forward.sort_values(by=['Player'])
        df_Forward=df_Forward.drop_duplicates(subset="Player",)

        if len(players_name)==0:
            a_values=df_Forward.iloc[0].values.tolist()    
            b_values=df_Forward.iloc[1].values.tolist()    
            a_values=a_values[3:]
            b_values=b_values[3:]

        elif len(players_name)==1:
            a_values=df_Forward.iloc[0].values.tolist()      
            a_values=a_values[3:]

        elif len(players_name)==2:
            a_values=df_Forward.iloc[0].values.tolist()    
            b_values=df_Forward.iloc[1].values.tolist()    
            a_values=a_values[3:]
            b_values=b_values[3:]

        elif len(players_name)==3:
            a_values=df_Forward.iloc[0].values.tolist()    
            b_values=df_Forward.iloc[1].values.tolist()    
            c_values=df_Forward.iloc[2].values.tolist()    
            a_values=a_values[3:]
            b_values=b_values[3:]
            c_values=c_values[3:]

        elif len(players_name)==4:
            a_values=df_Forward.iloc[0].values.tolist()    
            b_values=df_Forward.iloc[1].values.tolist()    
            c_values=df_Forward.iloc[2].values.tolist()    
            d_values=df_Forward.iloc[3].values.tolist()    
            a_values=a_values[3:]
            b_values=b_values[3:]
            c_values=c_values[3:]   
            d_values=d_values[3:]  

        elif len(players_name)==5:
            a_values=df_Forward.iloc[0].values.tolist()    
            b_values=df_Forward.iloc[1].values.tolist()    
            c_values=df_Forward.iloc[2].values.tolist()    
            d_values=df_Forward.iloc[3].values.tolist()    
            e_values=df_Forward.iloc[4].values.tolist()   
            a_values=a_values[3:]
            b_values=b_values[3:]
            c_values=c_values[3:]   
            d_values=d_values[3:]  
            e_values=e_values[3:]  

        params= list(df_Forward.columns)

        params= params[3:]
        #add ranges to list
        ranges=[]
        low = [] 
        high =[] 
        for x in params:
            a=min(df_Forward[params][x])
            a= a - (a * .25)

            b=max(df_Forward[params][x])
            b=b-(b*.25)     

            if a==b:
                b=b+0000.1

            low.append((a))
            high.append((b))

        # Add anything to this list where having a lower number is better
        # this flips the statistic
        lower_is_better = ['Miscontrol']

        radar = Radar(params, low, high,
                lower_is_better=lower_is_better,
                # whether to round any of the labels to integers instead of decimal places
                round_int=[False]*len(params),
                num_rings=4,  # the number of concentric circles (excluding center circle)
                # if the ring_width is more than the center_circle_radius then
                # the center circle radius will be wider than the width of the concentric circles
                ring_width=1, center_circle_radius=1)


        URL2 = ('https://raw.githubusercontent.com/googlefonts/SourceSerifProGFVersion/main/fonts/'
                'SourceSerifPro-ExtraLight.ttf')
        serif_extra_light = FontManager(URL2)
        URL3 = ('https://raw.githubusercontent.com/google/fonts/main/ofl/rubikmonoone/'
                'RubikMonoOne-Regular.ttf')
        rubik_regular = FontManager(URL3)
        URL4 = 'https://raw.githubusercontent.com/googlefonts/roboto/main/src/hinted/Roboto-Thin.ttf'
        robotto_thin = FontManager(URL4)
        URL5 = ('https://raw.githubusercontent.com/google/fonts/main/apache/robotoslab/'
                'RobotoSlab%5Bwght%5D.ttf')
        robotto_bold = FontManager(URL5)

        # plot radar
        fig, ax = radar.setup_axis()
        rings_inner = radar.draw_circles(ax=ax, facecolor='#ffb2b2', edgecolor='#fc5f5f')

        if len(players_name)==0:      
            radar1, vertices1 = radar.draw_radar_solid(a_values, ax=ax,
                                                    kwargs={'facecolor': '#aa65b2',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#502a54',
                                                            'lw': 3})

            radar2, vertices2 = radar.draw_radar_solid(b_values, ax=ax,
                                                    kwargs={'facecolor': '#66d8ba',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#216352',
                                                            'lw': 3})

            ax.scatter(vertices1[:, 0], vertices1[:, 1],
                    c='#aa65b2', edgecolors='#502a54', marker='o', s=150, zorder=2)
            ax.scatter(vertices2[:, 0], vertices2[:, 1],
                    c='#66d8ba', edgecolors='#216352', marker='o', s=150, zorder=2)   

        elif len(players_name)==1:      
            radar1, vertices1 = radar.draw_radar_solid(a_values, ax=ax,
                                                    kwargs={'facecolor': '#aa65b2',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#216352',
                                                            'lw': 3})

            ax.scatter(vertices1[:, 0], vertices1[:, 1],
                    c='#aa65b2', edgecolors='#502a54', marker='o', s=150, zorder=2)

        elif len(players_name)==2:      
            radar1, vertices1 = radar.draw_radar_solid(a_values, ax=ax,
                                                    kwargs={'facecolor': '#aa65b2',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#502a54',
                                                            'lw': 3})

            radar2, vertices2 = radar.draw_radar_solid(b_values, ax=ax,
                                                    kwargs={'facecolor': '#66d8ba',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#216352',
                                                            'lw': 3})

            ax.scatter(vertices1[:, 0], vertices1[:, 1],
                    c='#aa65b2', edgecolors='#502a54', marker='o', s=150, zorder=2)
            ax.scatter(vertices2[:, 0], vertices2[:, 1],
                    c='#66d8ba', edgecolors='#216352', marker='o', s=150, zorder=2)
        
        elif len(players_name)==3:      
            radar1, vertices1 = radar.draw_radar_solid(a_values, ax=ax,
                                                    kwargs={'facecolor': '#aa65b2',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#216354',
                                                            'lw': 3})

            radar2, vertices2 = radar.draw_radar_solid(b_values, ax=ax,
                                                    kwargs={'facecolor': '#66d8ba',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#216352',
                                                            'lw': 3})

            radar3, vertices3 = radar.draw_radar_solid(c_values, ax=ax,
                                                    kwargs={'facecolor': '#697cd4',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#222b54',
                                                            'lw': 3})     
            ax.scatter(vertices1[:, 0], vertices1[:, 1],
                    c='#aa65b2', edgecolors='#502a54', marker='o', s=150, zorder=2)
            ax.scatter(vertices2[:, 0], vertices2[:, 1],
                    c='#66d8ba', edgecolors='#216352', marker='o', s=150, zorder=2)
            ax.scatter(vertices3[:, 0], vertices3[:, 1],
                    c='#697cd4', edgecolors='#222b54', marker='o', s=150, zorder=2)                                                                                             

        elif len(players_name)==4:      
            radar1, vertices1 = radar.draw_radar_solid(a_values, ax=ax,
                                                    kwargs={'facecolor': '#aa65b2',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#216354',
                                                            'lw': 3})

            radar2, vertices2 = radar.draw_radar_solid(b_values, ax=ax,
                                                    kwargs={'facecolor': '#66d8ba',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#216352',
                                                            'lw': 3})

            radar3, vertices3 = radar.draw_radar_solid(c_values, ax=ax,
                                                    kwargs={'facecolor': '#697cd4',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#222b54',
                                                            'lw': 3})
            radar4, vertices4 = radar.draw_radar_solid(d_values, ax=ax,
                                                    kwargs={'facecolor': '#778821',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#4D5B05',
                                                            'lw': 3})   
                                                                                                                    
            ax.scatter(vertices1[:, 0], vertices1[:, 1],
                    c='#aa65b2', edgecolors='#502a54', marker='o', s=150, zorder=2)
            ax.scatter(vertices2[:, 0], vertices2[:, 1],
                    c='#66d8ba', edgecolors='#216352', marker='o', s=150, zorder=2)
            ax.scatter(vertices3[:, 0], vertices3[:, 1],
                    c='#697cd4', edgecolors='#222b54', marker='o', s=150, zorder=2)  
            ax.scatter(vertices4[:, 0], vertices4[:, 1],
                    c='#778821', edgecolors='#4D5B05', marker='o', s=150, zorder=2)      

        elif len(players_name)==5:      
            radar1, vertices1 = radar.draw_radar_solid(a_values, ax=ax,
                                                    kwargs={'facecolor': '#aa65b2',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#216354',
                                                            'lw': 3})

            radar2, vertices2 = radar.draw_radar_solid(b_values, ax=ax,
                                                    kwargs={'facecolor': '#66d8ba',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#216352',
                                                            'lw': 3})

            radar3, vertices3 = radar.draw_radar_solid(c_values, ax=ax,
                                                    kwargs={'facecolor': '#697cd4',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#222b54',
                                                            'lw': 3})
            radar4, vertices4 = radar.draw_radar_solid(d_values, ax=ax,
                                                    kwargs={'facecolor': '#778821',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#4D5B05',
                                                            'lw': 3})   
            radar5, vertices5 = radar.draw_radar_solid(e_values, ax=ax,
                                                    kwargs={'facecolor': '#a82228',
                                                            'alpha': 0.6,
                                                            'edgecolor': '#930e14',
                                                            'lw': 3})   

            ax.scatter(vertices1[:, 0], vertices1[:, 1],
                    c='#aa65b2', edgecolors='#502a54', marker='o', s=150, zorder=2)
            ax.scatter(vertices2[:, 0], vertices2[:, 1],
                    c='#66d8ba', edgecolors='#216352', marker='o', s=150, zorder=2)
            ax.scatter(vertices3[:, 0], vertices3[:, 1],
                    c='#697cd4', edgecolors='#222b54', marker='o', s=150, zorder=2)  
            ax.scatter(vertices4[:, 0], vertices4[:, 1],
                    c='#778821', edgecolors='#4D5B05', marker='o', s=150, zorder=2)   
            ax.scatter(vertices5[:, 0], vertices5[:, 1],
                    c='#a82228', edgecolors='#930e14', marker='o', s=150, zorder=2)   

        range_labels = radar.draw_range_labels(ax=ax, fontsize=20, fontproperties=robotto_bold.prop, color="#000000")
        param_labels = radar.draw_param_labels(ax=ax, fontsize=20, fontproperties=robotto_bold.prop, color="#000000")

        if len(players_name)==1:  
            fig_text(
                0.515, 0.99, cadena_nombre, size=22, fig=fig,
                highlight_textprops=[{"color": '#aa65b2'}],                                                                
                ha="center", fontproperties=robotto_bold.prop, color="#000000"
            )
        elif len(players_name)==2:  
                    fig_text(
                0.515, 0.99, cadena_nombre, size=22, fig=fig,
                highlight_textprops=[{"color": '#aa65b2'}, {"color": '#66d8ba'}],                                                                
                ha="center", fontproperties=robotto_bold.prop, color="#000000"
            )
        elif len(players_name)==3:  
                    fig_text(
                0.515, 0.99, cadena_nombre, size=22, fig=fig,
                highlight_textprops=[{"color": '#aa65b2'}, {"color": '#66d8ba'}, {"color": '#697cd4'}],                                                                
                ha="center", fontproperties=robotto_bold.prop, color="#000000"
            )
        elif len(players_name)==4:  
                    fig_text(
                0.515, 0.99, cadena_nombre, size=22, fig=fig,
                highlight_textprops=[{"color": '#aa65b2'}, {"color": '#66d8ba'}, {"color": '#697cd4'}, {"color": '#778821'}],                                                                
                ha="center", fontproperties=robotto_bold.prop, color="#000000"
            )
        elif len(players_name)==5:  
                    fig_text(
                0.515, 0.99, cadena_nombre, size=22, fig=fig,
                highlight_textprops=[{"color": '#aa65b2'}, {"color": '#66d8ba'}, {"color": '#697cd4'}, {"color": '#778821'}, {"color": '#a82228'}],                                                                
                ha="center", fontproperties=robotto_bold.prop, color="#000000"
            )
    
        # add subtitle
        fig.text(
            0.515, 0.942,
            "Los Angeles FC: Major League Soccer | Temporada 2023",
            size=20,
            ha="center", fontproperties=robotto_bold.prop, color="#000000"
        )



            # add credits
        CREDIT_1 = "data: vía fbref"
        CREDIT_2 = "inspired by: @everdata9"

        fig.text(
            0.99, 0.005, f"{CREDIT_1}\n{CREDIT_2}", size=12,
            fontproperties=robotto_bold.prop, color="#000000",
            ha="right"
        )
        

        st.write(fig)

    with col2:     

        st.header('****Tabla con datos comparativos****')       
        
        df1_transposed=df_Forward.iloc[:, 1:len(df_Forward.columns)]        
        df1_transposed.set_index('Player', inplace=True)
        df1_transposed = df1_transposed.T 
        df1_transposed = add_trunc(df1_transposed)
        st.table(df1_transposed)
    
       # CSS to inject contained in a string
        hide_table_row_index = """
                    <style>
                        table {
                        border-collapse: collapse;
                        width: 100%;
                        }

                        th, td {
                        text-align: left;
                        padding: 8px;
                        }

                        tr:nth-child(even) {
                        background-color: #f4f4f4;
                        }
                    </style>
                    """
    st.write(fechaAct)
    st.header('MÉTRICAS')
    st.markdown(text_contents)
    st.header('POSICIONES')
    st.markdown(txt_Position)




