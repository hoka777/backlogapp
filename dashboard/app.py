import streamlit as st
import pandas as pd
import io
import plotly.express as px
from utils import transform_backlog_to_summary, create_pivot,wrap_text,assign_tracks,working_hours_between

from openpyxl import load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import streamlit.components.v1 as components_html
import plotly.figure_factory as ff
from pandas.tseries.offsets import BusinessDay
import numpy as np

##############################################################################
# ---  Настройка страницы --- 
st.set_page_config(
    page_title="Мой дашборд", layout="wide", initial_sidebar_state="expanded")

##############################################################################
# --- Загрузка данных ---
st.sidebar.header("Загрузка файла")
uploaded_file = st.sidebar.file_uploader("Выберите Excel-файл", type=["xlsx", "xls"])
@st.cache_data
def load_data(file,sheet) -> pd.DataFrame:
    return pd.read_excel(file,sheet_name=sheet)

if not uploaded_file:
    st.sidebar.info("Загрузите файл для начала работы")
    st.stop()

xls = pd.ExcelFile(uploaded_file)
sheet_names = xls.sheet_names

# Показываем выпадающий список в сайдбаре
selected_sheet = st.sidebar.selectbox(
    "Выберите лист для основного DataFrame",
    options=sheet_names,
    index=0,                 # по умолчанию первый лист
    key="main_sheet"
)

##############################################################################
# --- Основные настройки --- 
# Изначальная загрузка данных и инициализация в session_state
st.session_state.df = load_data(uploaded_file, selected_sheet)
st.session_state.df_people = load_data(uploaded_file, 'ШТАТ(дашборд)')
st.session_state.df_sprint = load_data(uploaded_file, 'Спринты(дашборд)')
st.session_state.df_leave = load_data(uploaded_file, 'Отпуска(дашборд)')

df = st.session_state.df
df_sprint = st.session_state.df_sprint
df_people = st.session_state.df_people
df_leave = st.session_state.df_leave


# Кнопка для сброса изменений исходной таблицы
df_original = load_data(uploaded_file, selected_sheet)#'Лист1')
# def reset_data():
#     st.session_state.df = df_original.copy()
# st.sidebar.button("Сбросить исходную таблицу", on_click=reset_data)

##############################################################################
# --- Раздел 1: Исходная таблица ---
st.subheader("Исходная таблица")
# Настройка AgGrid
grid_df = GridOptionsBuilder.from_dataframe(df)
for col in df.columns:
    grid_df.configure_column(
        field=col, editable=True,
        filter="agSetColumnFilter", filterParams={'applyButton': True, 'clearButton': True},
        sortable=True, resizable=True,
        minWidth=10,       # минимальная ширина колонки
        maxWidth=200,      # максимальная ширина
        )
grid_df.configure_grid_options(domLayout='normal')

# Для конкретных столбцов можно жестко фиксировать ширину или flex
# gb.configure_column("Процесс (модуль)", width=200)
# gb.configure_column("ТРЗ", width=100)
# gb.configure_column("Роль", flex=1)   # flex=1 — займет всё оставшееся пространство

grid_options = grid_df.build()
# Отображение AgGrid с enterprise фичами (для Set Filter)
grid_response = AgGrid(
    df,
    gridOptions=grid_options,
    enable_enterprise_modules=True,
    update_mode=GridUpdateMode.MODEL_CHANGED,
    fit_columns_on_grid_load=False,
    height=500)
# Обновление данных после редактирования
st.session_state.df = pd.DataFrame(grid_response['data'])
df = st.session_state.df

st.error("! Выберете колонку, где находятся названия задач!")
column_task_name  = st.selectbox("Названия задач брать из колонки", df.columns.unique().to_list(), index=8)

##############################################################################
# --- Раздел 1.1: Калькулятор рабочих часов ---
column1, column2,column3 = st.columns(3)
with column1:
    start_date = st.date_input("Дата начала", value=pd.to_datetime("today").date())
with column3:
    end_date = st.date_input("Дата окончания", value=pd.to_datetime("today").date())
    if start_date > end_date:
        st.error("❗ Дата начала должна быть раньше или равна дате окончания.")
    else:
        with column1:
            hours_per_day = st.number_input("Рабочих часов в дне", min_value=1, max_value=24, value=8)
        total_hours = working_hours_between(start_date, end_date, hours_per_day)
        with column2:
            # st.write("### Статистика по кварталам")
            # st.write("В Q3 ", working_hours_between('2025-06-30', '2025-08-31', 8), 'рабочих часов')
            # st.write("В Q4 ", working_hours_between('2025-09-01', '2025-12-31', 8), 'рабочих часов')
            st.success(f"Между {start_date} и {end_date} включая оба дня:\n"
                        f"• рабочих дней: {total_hours // hours_per_day}\n"
                        f"• рабочих часов: {total_hours}")


##############################################################################
# --- Раздел 2: Подготовка Gantt ---
# st.subheader("Таблица для Ганта")
st.session_state.df_gantt = transform_backlog_to_summary(df,df_sprint,column_task_name)
df_g = st.session_state.df_gantt
# st.write(df_g)
df_g = pd.merge(
    st.session_state.df_gantt,
    df_sprint[['Номер спринта', 'Дата начала', 'Дата окончания']], on="Номер спринта",how="left")
# # Конвертируем в формат date (без времени)
# df_g['ТРЗ'] = df_g['ТРЗ'].fillna(0).astype(int)
# df_g['Дата начала']   = pd.to_datetime(df_g['Дата начала'])#.dt.strftime('%Y-%m-%d')
# df_g["Дата окончания"] = df_g.apply(lambda row: row["Дата начала"] + BusinessDay(row["ТРЗ"]),axis=1)

# df_g['Дата начала']   = pd.to_datetime(df_g['Дата начала']).dt.strftime('%Y-%m-%d')                                                 
# df_g['Дата окончания'] = pd.to_datetime(df_g['Дата окончания']).dt.strftime('%Y-%m-%d')

# 1) Переименуем колонки спринта, чтобы не путаться
df_g.rename(columns={
                'Дата начала': 'Дата начала спринта',
                'Дата окончания': 'Дата окончания спринта'},inplace=True)

df_g['Sprint_Start'] = df_g['Дата начала спринта']
df_g['Sprint_End'] = df_g['Дата окончания спринта']

# Приводим Sprint_Start в datetime и заполняем ТРЗ
df_g['Sprint_Start'] = pd.to_datetime(df_g['Sprint_Start'])
df_g['ТРЗ'] = df_g['ТРЗ'].fillna(1).astype(int)

# 1) Считаем предварительные начала/концы:
#    - для Аналитик/UX/UI они стартуют сразу в Sprint_Start
#    - все остальные (включая QA) пока считаем стартующими после Sprint_Start
df_g['Start_calc'] = df_g['Sprint_Start']
df_g['Finish_calc'] = df_g['Start_calc'] + df_g['ТРЗ'].apply(BusinessDay)

# 2) Определяем момент, когда заканчиваются Аналитик+UX/UI (AD_finish):
task_keys = ['Процесс (модуль)','Направление','Название крупная задача (ЭПИК)','Номер спринта']
mask_ad = df_g['Роль'].isin(['Эксперт RnD','Аналитик','Архитектор'])
ad_finish = (df_g[mask_ad]
             .groupby(task_keys)['Finish_calc']
             .max()
             .reset_index()
             .rename(columns={'Finish_calc':'AD_finish'}))
df_g = df_g.merge(ad_finish, on=task_keys, how='left')

# 3) Пересчитаем для всех ролей (кроме AD) их реальные Start/Finish относительно AD_finish:
#    - Аналитик/UX/UI: уже правильно
#    - остальные кроме QA: стартуют из AD_finish
mask_non_ad = ~df_g['Роль'].isin(['Эксперт RnD','Аналитик','Архитектор','UX/UI','QA'])
df_g.loc[mask_non_ad, 'Start_calc'] = df_g.loc[mask_non_ad, 'AD_finish']
df_g.loc[mask_non_ad, 'Finish_calc'] = (
    df_g.loc[mask_non_ad, 'Start_calc']+ df_g.loc[mask_non_ad, 'ТРЗ'].apply(BusinessDay)
)

# 4) Вычисляем, когда все НЕ-QA роли завершатся:
nonqa_finish = (
    df_g[~df_g['Роль'].eq('QA')]
    .groupby(task_keys)['Finish_calc']
    .max()
    .reset_index()
    .rename(columns={'Finish_calc':'NonQA_finish'})
)
df_g = df_g.merge(nonqa_finish, on=task_keys, how='left')

# 5) Наконец — для QA: стартуем после NonQA_finish и считаем свой Finish:
mask_qa = df_g['Роль'].eq('QA')
df_g.loc[mask_qa, 'Start_calc'] = df_g.loc[mask_qa, 'NonQA_finish']
df_g.loc[mask_qa, 'Finish_calc'] = (
    df_g.loc[mask_qa, 'Start_calc']+ df_g.loc[mask_qa, 'ТРЗ'].apply(BusinessDay)
)
# st.write(df_g)

# 6) Переименуем в понятные столбцы и отформатируем для AgGrid:
df_g['Дата начала']   = df_g['Start_calc']#.dt.strftime('%Y-%m-%d')
df_g['Дата окончания'] = df_g['Finish_calc']#.dt.strftime('%Y-%m-%d')

# Настройка AgGrid для df_g
# grid_gantt = GridOptionsBuilder.from_dataframe(df_g)
# for col in df_g.columns:
#     grid_gantt.configure_column(
#         field=col,
#         editable=True,
#         filter="agSetColumnFilter",
#         filterParams={'applyButton': True, 'clearButton': True},
#         sortable=True,
#         resizable=True)
# grid_gantt.configure_grid_options(domLayout='normal')
# grid_options = grid_gantt.build()
# grid_response1 = AgGrid(
#     df_g,
#     gridOptions=grid_options,
#     enable_enterprise_modules=True,
#     update_mode=GridUpdateMode.MODEL_CHANGED, 
#     fit_columns_on_grid_load=True           # подгонка колонок
# )

# # сохраняем обратно в session_state
# st.session_state.df_gantt = pd.DataFrame(grid_response1['data'])
# df_g = st.session_state.df_gantt
# st.data_editor(df_g, use_container_width=False)


##############################################################################
# --- Настройка и отображение Gantt-диаграммы --- 
st.subheader("Настройка Gantt-диаграммы и графика")
col1, col2,col3 = st.columns([1, 1,1])  # или st.columns(2)
with col1:
    st.write("Фильтры")
    cols = df_g.columns.tolist()
    # выбор колонок
    task_col  = st.selectbox("Подпись в тултипе", cols, index=3)
    y_axis   = st.selectbox("Группировать по (ось Y)", cols, index=2)
    color_by = st.selectbox("Окрашивать по", cols, index=0)

    start_col = "Дата начала" #st.selectbox("Столбец с датой начала", cols, index=1)
    end_col   = "Дата окончания" #st.selectbox("Столбец с датой окончания", cols, index=2)

with col2:
    max_len = st.slider("Длина подписей", min_value=10, max_value=100, value=80)
    font_size = st.slider("Размер текста", min_value=1, max_value=50, value=8)
    width_plot = st.slider("Ширина диаграммы", min_value=500, max_value=2000, value=1700)
    height_plot = st.slider("Высота диаграммы", min_value=500, max_value=2500, value=1200)
with col3:
    swap_text =st.checkbox("Включить перенос слов",value=True)
    bar_mode = st.checkbox("Разделять бары")
    sort_val_first = st.selectbox("Сортировать задачи по", ['Task','Start','Finish','Y_Group','Resource'],index=1)
    # sort_val_second = st.selectbox("Затем по", ['Task','Start','Finish','Y_Group','Resource'])
    asc_desc = st.selectbox("Направление", ['ASC','DESC'])

    
selected_module = st.multiselect("Модули", df_g['Процесс (модуль)'].unique(), default=df_g['Процесс (модуль)'].unique())
df_g = df_g[df_g['Процесс (модуль)'].isin(selected_module)]
# подготовка данных
gantt_df = df_g[[task_col, start_col, end_col, y_axis, color_by]].copy()
gantt_df[start_col] = pd.to_datetime(gantt_df[start_col], errors='coerce')
gantt_df[end_col]   = pd.to_datetime(gantt_df[end_col], errors='coerce')
gantt_df.columns = ["Task", "Start", "Finish", "Y_Group", "Resource"]


if swap_text:
    gantt_df['Task'] = gantt_df['Task'].apply(lambda s: wrap_text(s, max_len=max_len))
    gantt_df['Y_Group'] = gantt_df['Y_Group'].apply(lambda s: wrap_text(s, max_len=max_len))
    gantt_df['Resource'] = gantt_df['Resource'].apply(lambda s: wrap_text(s, max_len=max_len))

##############################################################################
# st.write(gantt_df)
# --- Раздел 3: Gantt-диаграмма ---


def plot_gantt(gantt_d,bar_mode = True):
    y_group = "Y_Group"
    gantt_df = gantt_d
    if bar_mode:
        gantt_df = assign_tracks(gantt_df)
        # 3. Делаем новую категорию, объединяя имя группы и номер трека
        gantt_df['Y_Group_lane'] = gantt_df['Y_Group'] + ' (lane ' + (gantt_df['track']+1).astype(str) + ')'
        gantt_df['number_lane'] = (gantt_df['track']+1).astype(str)
        gantt_df.sort_values(['Y_Group','number_lane'], ascending=False,inplace=True)
        y_group="Y_Group_lane"
    
    # отображение Plotly Gantt
    fig = px.timeline(
        gantt_df,
        x_start="Start",x_end="Finish",
        y=y_group, color="Resource",
        title="График занятости", text="Task",
        opacity=0.6, template='plotly',
        )
    fig.update_traces(
        textposition="inside",
        insidetextanchor="end",
        textangle=0,
        textfont=dict(size=25),
        width=0.9,               # чем больше тем толще
        offsetgroup=0,
    )
    
    x_min = pd.to_datetime(df_g['Дата начала']).min()
    # смещаем назад на number_of_days = weekday, чтобы попасть на понедельник той же недели
    first_monday = x_min - pd.to_timedelta(x_min.weekday(), unit="d")
    fig.update_xaxes(
        showgrid=True,                  # включить основные gridlines
        gridwidth=1,                    
        gridcolor="rgba(200,200,200,0)",
        tickformat="%d.%m",             # формат подписи дат
        tick0=first_monday.strftime("%Y-%m-%d"),
        dtick=7 * 24 * 60 * 60 * 1000,                     # основной шаг: 1 день
        # minor=dict(
        #     showgrid=False,              # включить минорные gridlines
        #     gridwidth=1,
        #     gridcolor="rgba(200,200,200,0.25)",
        #     dtick="12h"                 # шаг минорной сетки: 12 часов
        # )
    )
    y_cats = gantt_df[y_group].unique().tolist()
    fig.update_yaxes(
        automargin=True,
        tickfont=dict(size=font_size),
        categoryorder="array",
        categoryarray=y_cats,
        autorange="reversed",
        )
    

    # Уменьшить зазор между дорожками (0 = вообще нет промежутка)
    fig.update_layout(
        font=dict(color='white'),
        # bargap=1,              # уменьшить пустое пространство между категориями
        # bargroupgap=1,           # убрать зазор между группами, если они есть
        height=height_plot ,              # при желании увеличить общую высоту графика
        barmode="overlay",  # установить группировку по Y
        paper_bgcolor='rgba(0,0,0,0)',  # фон всего холста
        plot_bgcolor='rgba(0,0,0,0)',   # фон области графика
        # legend=dict(
        #     orientation="v",     # вертикальное (по умолчанию)
        #     yanchor="top",
        #     y=0.95,              # чуть ниже верхней границы
        #     xanchor="right",
        #     x=0.95)
    )



    # Фоновые полосы спринтов
    for _, sprint in st.session_state.df_sprint.iterrows():
        sprint_start = pd.to_datetime(sprint['Дата начала'], errors='coerce')
        sprint_end = pd.to_datetime(sprint['Дата окончания'], errors='coerce')
        sprint_num = sprint['Номер спринта']
        if pd.notnull(sprint_start) and pd.notnull(sprint_end):
            fig.add_vrect(
                x0=sprint_start,
                x1=sprint_end,
                fillcolor="LightSalmon",
                opacity=0.3,
                layer="below",
                line_width=0,
                
            )
            # единая аннотация с переносом строки: номер спринта и даты
            mid = sprint_start + (sprint_end - sprint_start) / 2
            fig.add_annotation(
                x=mid,
                y=1.04,
                xref="x",
                yref="paper",
                text=(f"Спринт {sprint_num}<br>{sprint_start.date():%d.%m.%Y}<br>{sprint_end.date():%d.%m.%Y}"),
                showarrow=False,            
                font=dict(size=font_size, color="White"),
                align="center"      
            )

    # Автоматическое горизонтальное выделение каждой категории Y
    # Вычисляем диапазон X по данным Gantt
    x_min = gantt_df['Start'].min()
    x_max = gantt_df['Finish'].max()
    for idx, cat in enumerate(y_cats):
        # отступ вверх и вниз по оси Y для полосы
        y0 = idx - 0.4
        y1 = idx + 0.4
        fig.add_shape(
            type="rect",
            x0=x_min, x1=x_max,
            y0=y0, y1=y1,
            xref="x", yref="y",
            fillcolor="LightBlue" if idx % 2 == 0 else "LightGray",
            opacity=0.1,
            layer="below",
            line_width=0
        )
    # Для горизонтального скролла и фиксированной ширины вставляем через raw HTML
    chart_html = fig.to_html(include_plotlyjs='cdn', full_html=False)
    components_html.html(
        f"""
        <div style='width:100%; overflow-x:auto;background:rgba(200,200,200,0.7);'>
        <div style='min-width:{width_plot}px;background:rgba(200,200,200,0.7);'>
            {chart_html}
        </div>
        </div>
        """,
        height=height_plot
    )

if asc_desc == "ASC":
    asc_desc=True
else:
    asc_desc=False
plot_gantt(gantt_df.sort_values(by=[sort_val_first],ascending=asc_desc),bar_mode)


# # Раздел 3: Сводная таблица
# st.subheader("Настройка и просмотр сводной таблицы")
# all_cols = df.columns.tolist()
# index_cols   = st.multiselect("Выберите индексные столбцы", all_cols)
# columns_cols = st.multiselect("Выберите столбцы для колонок", all_cols)
# value_col    = st.selectbox("Выберите столбец для значений", all_cols)
# aggfunc      = st.selectbox("Функция агрегации", ["sum", "mean", "count", "min", "max"], index=0)
# if index_cols and value_col:
#     pivot = create_pivot(df, index_cols, columns_cols, value_col, aggfunc)
#     if pivot is not None:
#         st.dataframe(pivot, use_container_width=True)
# else:
#     st.info("Выберите хотя бы индексные столбцы и столбец значений.")





# # --- Раздел 4: Фильтры для анализа нагрузки ---
# st.subheader("Фильтры для анализа нагрузки")
# selected_people = st.multiselect("Исполнители", df_g['Исполнитель'].unique(), default=df_g['Исполнитель'].unique())
# selected_roles  = st.multiselect("Роли", df_g['Роль'].unique(), default=df_g['Роль'].unique())
# selected_sprints= st.multiselect("Спринты", df_g['Номер спринта'].unique(), default=df_g['Номер спринта'].dropna().unique())

# df_f = df_g[
#     df_g['Исполнитель'].isin(selected_people) &
#     df_g['Роль'].isin(selected_roles) &
#     df_g['Номер спринта'].isin(selected_sprints)
# ].copy()

df_f = df_g.dropna(subset=['Дата начала', 'Дата окончания']).copy()

col_start, col_end = st.columns(2)
with col_start:
    period_start = st.date_input(
        "Период: начало",
        value=pd.to_datetime(df_f['Дата начала'].min()).date()
    )
with col_end:
    period_end   = st.date_input(
        "Период: конец",
        value=pd.to_datetime(df_f['Дата окончания'].max()).date()
    )
df_f['Дата начала'] = pd.to_datetime(df_f['Дата начала'], errors='coerce')
df_f['Дата окончания'] = pd.to_datetime(df_f['Дата окончания'], errors='coerce')
df_f = df_f[(df_f['Дата начала'].dt.date >= period_start) &
            (df_f['Дата окончания'].dt.date <= period_end)]
        

# # --- Раздел 5: Сводная по ресурсам --- скорей всего неправильно считаются дни
# st.subheader("Сводная по ресурсам")
# records = []
# for person in df_f['Исполнитель'].unique():
#     # всего раб. дней
#     sprints = df_f[df_f['Исполнитель']==person]['Номер спринта'].unique()
#     total_wd = 0
#     for s in sprints:
#         sp = df_sprint[df_sprint['Номер спринта']==s]
#         a = pd.to_datetime(sp['Дата начала'].iloc[0]).date()
#         b = pd.to_datetime(sp['Дата окончания'].iloc[0]).date()
#         total_wd += np.busday_count(a, b + pd.Timedelta(days=1))
#     occupied = df_f[df_f['Исполнитель']==person]['ТРЗ'].sum()
#     # отпуск из df_leave
#     vac = 0
#     for _, r in df_leave[df_leave['Исполнитель']==person].iterrows():
#         a = pd.to_datetime(r['Начало']).date()
#         b = pd.to_datetime(r['Конец']).date()
#         vac += np.busday_count(a, b + pd.Timedelta(days=1))
#     records.append({
#         'Исполнитель': person,
#         'Роль': df_f[df_f['Исполнитель']==person]['Роль'].mode()[0],
#         'Всего раб. дн.': total_wd,
#         'Занято (ТРЗ дн.)': occupied,
#         'Дн. отпуска': vac,
#         'Доступно (дн.)': total_wd - occupied - vac
#     })

# df_res = pd.DataFrame(records)
# gb_res = GridOptionsBuilder.from_dataframe(df_res)
# gb_res.configure_default_column(resizable=True, sortable=True)
# AgGrid(df_res, gridOptions=gb_res.build(), fit_columns_on_grid_load=True, height=300)



# 5) Вычисление свободной ёмкости по периодам с учётом отпусков
st.subheader("Свободная ёмкость исполнителей в период")

records_cap = []
for p in df_f['Исполнитель'].unique():
    # всего рабочих дней в выбранном периоде
    total_bd = np.busday_count(period_start, period_end + pd.Timedelta(days=1))
    # считаем дни отпуска пересечением с периодом
    vac_days = 0
    for _, v in df_leave[df_leave['Исполнитель'] == p].iterrows():
        vs = pd.to_datetime(v['Начало']).date()
        ve = pd.to_datetime(v['Конец']).date()
        start_int = max(vs, period_start)
        end_int   = min(ve, period_end)
        if start_int <= end_int:
            vac_days += np.busday_count(start_int, end_int + pd.Timedelta(days=1))
    # суммарная нагрузка по задачам (ТРЗ в днях)
    task_bd = df_f[df_f['Исполнитель'] == p]['ТРЗ'].sum()
    free_bd = total_bd - vac_days - task_bd

    records_cap.append({
        'Исполнитель': p,
        'Всего раб. дн.': total_bd,
        'Дн. отпуска': vac_days,
        'Занято (ТРЗ дн.)': task_bd,
        'Доступно (дн.)': free_bd
    })

df_res = pd.DataFrame(records_cap)

# Выводим в AgGrid
gb_cap = GridOptionsBuilder.from_dataframe(df_res)
gb_cap.configure_default_column(resizable=True, sortable=True)
AgGrid(
    df_res,
    gridOptions=gb_cap.build(),
    fit_columns_on_grid_load=True,
    height=300,
    key="capacity_aggrid"
)

# st.write(df_res.sort_values('Доступно (дн.)'))

#  Горизонтальная диаграмма «Нагрузка vs Отпуск vs Свободно»
fig = px.bar(
    df_res.sort_values('Доступно (дн.)'),
    y='Исполнитель',
    x=['Занято (ТРЗ дн.)','Дн. отпуска','Доступно (дн.)'],
    orientation='h',
    title="Нагрузка vs Отпуск vs Свободное время",
    labels={'value':'Дней','variable':'Категория'}
)
bar_px=30
fig.update_layout(barmode='stack', height=len(df_res) * bar_px)
st.plotly_chart(fig, use_container_width=True)

# Стековая диаграмма объёма задач по ролям
pivot = (
    df_f
    .pivot_table(
        index='Роль',
        columns='Исполнитель',
        values='ТРЗ',
        aggfunc='sum',
        fill_value=0
    )
    .reset_index()
)
fig = px.bar(
    pivot,
    x='Роль',
    y=pivot.columns.drop('Роль'),
    title="Объём задач по ролям и исполнителям"
)
fig.update_layout(barmode='stack', height=400, xaxis_title=None)
st.plotly_chart(fig, use_container_width=True)


# Тепловая карта ежедневной загрузки и отпуска
import plotly.express as px

# Сформируем матрицу: index=Исполнитель, cols=даты спринта
df_f = df_f.dropna()
dates = pd.date_range(
    df_f['Дата начала'].dropna().min(),
    df_f['Дата окончания'].dropna().max(),
    freq='D'
)
persons = df_f['Исполнитель'].unique()
cal = pd.DataFrame(0, index=persons, columns=dates)

# Нагрузка
for _, r in df_f.iterrows():
    p = r['Исполнитель']
    start, end = pd.to_datetime(r['Дата начала']), pd.to_datetime(r['Дата окончания'])
    for d in pd.date_range(start, end, freq='D'):
        if d.weekday()<5:
            cal.at[p, d] += 1

# Отпуска пометим -1
for _, r in df_leave.iterrows():
    p = r['Исполнитель']
    for d in pd.date_range(r['Начало'], r['Конец'], freq='D'):
        if d.weekday()<5 and p in cal.index and d in cal.columns:
            cal.at[p, d] = -1

fig = px.imshow(
    cal,
    labels=dict(x="Дата", y="Исполнитель", color="Нагрузка"),
    x=cal.columns,
    y=cal.index,
    title="Ежедневная загрузка (и отпуска)"
)
# Перекрашиваем отпуск в серый:
fig.update_traces(
    zmin=-1, zmax=cal.values.max(),
    colorscale=[
        [0.0, "lightgray"], [0.01, "white"],
        [0.5, "lightblue"], [1.0, "blue"]
    ]
)
st.plotly_chart(fig, use_container_width=True, height=500)

# Объём задач по спринтам
sprint_vol = (
    df_f
    .groupby('Номер спринта')['ТРЗ']
    .sum()
    .reset_index(name='Объём TRЗ')
)
fig = px.bar(
    sprint_vol,
    x='Номер спринта',
    y='Объём TRЗ',
    title="Объём задач по спринтам"
)
st.plotly_chart(fig, use_container_width=True, height=350)

# 6. Таблица конфликтов задач с отпусками
conflicts = []
for _, r in df_f.iterrows():
    p = r['Исполнитель']
    s, e = pd.to_datetime(r['Дата начала']), pd.to_datetime(r['Дата окончания'])
    leaves = df_leave[df_leave['Исполнитель']==p]
    if any(not (e < ld or s > lu)
           for ld, lu in zip(leaves['Начало'], leaves['Конец'])):
        conflicts.append(r)

df_conf = pd.DataFrame(conflicts)
st.subheader("Задачи, пересекающиеся с отпусками")
if not df_conf.empty:
    gb2 = GridOptionsBuilder.from_dataframe(df_conf)
    gb2.configure_default_column(resizable=True, sortable=True)
    AgGrid(df_conf, gridOptions=gb2.build(), height=300)
else:
    st.success("Конфликтов не обнаружено")
