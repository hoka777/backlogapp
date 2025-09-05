import streamlit as st
import pandas as pd
# import io
import plotly.express as px
from utils import transform_backlog_to_summary, \
        create_pivot,\
            wrap_text,\
                assign_tracks,\
                    working_hours_between,\
                    transform_gantt

from openpyxl import load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import streamlit.components.v1 as components_html
import plotly.figure_factory as ff
from pandas.tseries.offsets import BusinessDay
import numpy as np
from functools import partial
AgGrid = partial(AgGrid, allow_unsafe_jscode=True)
from st_aggrid.shared import JsCode
from plot_graph import plot_gantt, ui_gantt_settings,ui_theme_picker

import streamlit as st



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
    "Выберите лист где находится бэклог",
    options=sheet_names,
    index=6,                 # по умолчанию первый лист
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
st.info('! Здесь отображена исходная таблица с выбранного листа, отфильтруйте ее по нужным колонкам - далее в расчетах будет использоваться именно эта таблица. Вы также можете менять значения в таблице и сразу увидеть изменения (они не повлияют на ваш исходный файл)')
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
    # update_mode=GridUpdateMode.MODEL_CHANGED,
    update_mode=(
        GridUpdateMode.MODEL_CHANGED            # редактирование ячеек
        | GridUpdateMode.FILTERING_CHANGED      # фильтрация
        | GridUpdateMode.SORTING_CHANGED        # сортировка
    ),
    fit_columns_on_grid_load=False,
    height=500)
# Обновление данных после редактирования
st.session_state.df = pd.DataFrame(grid_response['data'])
df = st.session_state.df

with st.expander("Выбор исходных данных", expanded=False):
    st.error("! Выберете названия колонок, где находятся названия эпиков и декомпозированных задач соответсвенно!")
    col1, col2 = st.columns(2)
    with col1:
        column_epic_name  = st.selectbox("Эпики", df.columns.unique().to_list(), index=10)
    with col2:
        column_task_name  = st.selectbox("Задачи", df.columns.unique().to_list(), index=13)
st.write('  ')
st.write('  ')
with st.expander("Режим работы", expanded=False):
    st.error("! Выберете режим отображения эпиков или задач! В режиме эпиков будут отображаться названия эпиков, при этом ТРЗ будут суммировать по всем задачам входящих в Эпик. В режиме Задача есть возможность анализировать отдельно задачи, входящие в эпик")
    mode = st.radio(
        "Выберите режим:",
        ["Эпики", "Задачи"],
        index=0,   # что выбрано по умолчанию
        horizontal=True # в одну строку
    )
print('here')
# st.write("Текущий режим:", mode)

##############################################################################
# --- Раздел 1.1: Калькулятор рабочих часов ---
# column1, column2,column3 = st.columns(3)
# with column1:
#     start_date = st.date_input("Дата начала", value=pd.to_datetime("today").date())
# with column3:
#     end_date = st.date_input("Дата окончания", value=pd.to_datetime("today").date())
#     if start_date > end_date:
#         st.error("❗ Дата начала должна быть раньше или равна дате окончания.")
#     else:
#         with column1:
#             hours_per_day = st.number_input("Рабочих часов в дне", min_value=1, max_value=24, value=8)
#         total_hours = working_hours_between(start_date, end_date, hours_per_day)
#         with column2:
#             # st.write("### Статистика по кварталам")
#             # st.write("В Q3 ", working_hours_between('2025-06-30', '2025-08-31', 8), 'рабочих часов')
#             # st.write("В Q4 ", working_hours_between('2025-09-01', '2025-12-31', 8), 'рабочих часов')
#             st.success(f"Между {start_date} и {end_date} включая оба дня:\n"
#                         f"• рабочих дней: {total_hours // hours_per_day}\n"
#                         f"• рабочих часов: {total_hours}")


##############################################################################
# --- Раздел 2: Подготовка Gantt ---
# st.subheader("Таблица для Ганта")
st.session_state.df_gantt = transform_backlog_to_summary(df,df_sprint,column_epic_name,column_task_name,mode)
df_g = st.session_state.df_gantt
# st.write(df_g)

df_g = pd.merge(
    st.session_state.df_gantt,
    df_sprint[['Номер спринта', 'Дата начала', 'Дата окончания']], on="Номер спринта",how="left")

# 1) Переименуем колонки спринта, чтобы не путаться
df_g.rename(columns={
                'Дата начала': 'Дата начала спринта',
                'Дата окончания': 'Дата окончания спринта'},inplace=True)

df_g['Sprint_Start'] = df_g['Дата начала спринта']
df_g['Sprint_End'] = df_g['Дата окончания спринта']

# Приводим Sprint_Start в datetime и заполняем ТРЗ
df_g['Sprint_Start'] = pd.to_datetime(df_g['Sprint_Start'])
df_g["ТРЗ"] = (
    pd.to_numeric(df_g["ТРЗ"], errors="coerce")   # всё, что нельзя в число -> NaN
      .fillna(1)                                  # заменяем NaN на 1
      .astype(int)                                # теперь можно в int
)

# df_g['ТРЗ'] = df_g['ТРЗ'].fillna(1).astype(int)

# 1) Считаем предварительные начала/концы:
#    - для Аналитик/UX/UI они стартуют сразу в Sprint_Start
#    - все остальные (включая QA) пока считаем стартующими после Sprint_Start
df_g['Start_calc'] = df_g['Sprint_Start']
df_g['Finish_calc'] = df_g['Start_calc'] + df_g['ТРЗ'].apply(BusinessDay)

# 2) Определяем момент, когда заканчиваются Аналитик+UX/UI (AD_finish):
task_keys = ['Feature (модуль)','Направление','Название крупная задача (ЭПИК)','Номер спринта']
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



#############################################################################
# --- Настройка и отображение Gantt-диаграммы --- 
st.subheader("Gantt-диаграмма поэтапная")
st.info("Данная диаграмма учитывает последовательность разработки - сначала Аналитик/UX/UI/Архитектор (по макс времени), потом роли разработки (также по макс) и затем QA.")
settings_gant1 = ui_gantt_settings(df_g, prefix="plotly", title="⚙️ Настройки",default_y_idx=0,default_task_idx=1,default_color_idx=2)
# подготовка данных
gantt_df = df_g[[settings_gant1['task_col'], settings_gant1['start_col'], settings_gant1['end_col'], settings_gant1['y_axis'], settings_gant1['color_by']]].copy()
gantt_df[settings_gant1['start_col']] = pd.to_datetime(gantt_df[settings_gant1['start_col']], errors='coerce')
gantt_df[settings_gant1['end_col']]   = pd.to_datetime(gantt_df[settings_gant1['end_col']], errors='coerce')
gantt_df.columns = ["Task", "Start", "Finish", "Y_Group", "Resource"]
# if settings_gant1['swap_text']:
#     gantt_df['Task'] = gantt_df['Task'].apply(lambda s: wrap_text(s, max_len=settings_gant1['max_len']))
#     gantt_df['Y_Group'] = gantt_df['Y_Group'].apply(lambda s: wrap_text(s, max_len=settings_gant1['max_len']))
#     gantt_df['Resource'] = gantt_df['Resource'].apply(lambda s: wrap_text(s, max_len=settings_gant1['max_len']))
if settings_gant1['asc_desc'] == "ASC":
    asc_desc=True
else:
    asc_desc=False  
theme_key, theme, template_name = ui_theme_picker(expanded=False, default_key="pastel",)
with st.expander("График", expanded=True):
    plot_gantt(gantt_df.sort_values(by=[settings_gant1['sort_val_first']],ascending=asc_desc),
                #    start_column="Start",#settings_gant2["start_col"],
                #    end_column="Finish",#settings_gant2["end_col"],
                #    y_group=settings_gant1['y_axis'],
                #    color_column=settings_gant1["color_by"],
                #    text_column=settings_gant1["task_col"],
                graph_title="График занятости",
                font_size=settings_gant1["font_size"],
                width_plot=settings_gant1["width_plot"],
                height_plot =settings_gant1["height_plot"],
                fit_names=settings_gant1['fit_names'],
                font_size_names=settings_gant1['font_size_names'],
                max_len=settings_gant1['max_len'],
                swap_text=settings_gant1['swap_text'],
                theme=theme,
                sprint_df=st.session_state.df_sprint,
                )

##########################################################
st.subheader("Gantt-диаграмма компактная")
st.write("Данный график не учитывает последовательность разработки - по эпику суммируется ТРЗ по всем ролям.")
lane_df = transform_gantt(gantt_df)
settings_gant2 = ui_gantt_settings(lane_df, prefix="plotly1", 
                                   title="⚙️ Настройки",
                                   default_y_idx = 5, 
                                   default_task_idx = 0,
                                   default_color_idx = 4)
theme_key1, theme1, template_name1 = ui_theme_picker(expanded=False, default_key="pastel",suffix="1")
with st.expander("График", expanded=True):
    # st.write(settings_gant2)
    plot_gantt(lane_df,
                start_column="Start",#settings_gant2["start_col"],
                end_column="Finish",#settings_gant2["end_col"],
                y_group=settings_gant2['y_axis'],
                color_column=settings_gant2["color_by"],
                text_column=settings_gant2["task_col"],
                graph_title="График занятости",
                font_size=settings_gant2["font_size"],
                width_plot=settings_gant2["width_plot"],
                height_plot =settings_gant2["height_plot"],
                fit_names=settings_gant2['fit_names'],
                font_size_names=settings_gant2['font_size_names'],
                max_len=settings_gant2['max_len'],
                swap_text=settings_gant2['swap_text'],
                theme=theme1,
                sprint_df=st.session_state.df_sprint,
                )

############################################################
# ===== Сводная таблица ТРЗ по стекам и общая =====
roles = ['Эксперт RnD','Аналитик','Архитектор','UX/UI','C#','Py','React','QA']
trz_cols = [f"трз {r}" for r in roles if f"трз {r}" in df.columns]
if trz_cols:
    trz_sum = df[trz_cols].apply(pd.to_numeric, errors='coerce').fillna(0).sum()
    total_trz = trz_sum.sum()
    summary_df = pd.DataFrame([trz_sum])
    summary_df.index = ["Сумма ТРЗ по всем задачам"]
    summary_df["Итого"] = total_trz
    st.subheader("Сумма ТРЗ по всем задачам (по стекам и общая)")
    st.dataframe(summary_df)


# -------- Свод по задачам (Эпик/column_task_name): сумма ТРЗ по стекам и Итого --------
try:
    _roles = ['Эксперт RnD','Аналитик','Архитектор','UX/UI','C#','Py','React','QA']
    _trz_cols = [f"трз {r}" for r in _roles if f"трз {r}" in df.columns]
    if _trz_cols:
        _num = df[_trz_cols].replace(',', '.', regex=True).apply(pd.to_numeric, errors='coerce').fillna(0)
        _tmp = df[[column_epic_name]].copy()
        _tmp[_trz_cols] = _num
        task_sum = _tmp.groupby(column_epic_name, dropna=False)[_trz_cols].sum().reset_index()
        task_sum["Итого"] = task_sum[_trz_cols].sum(axis=1)
        task_sum = task_sum.sort_values("Итого", ascending=False)
        task_sum.insert(1, "Итого", task_sum.pop("Итого"))
        st.subheader("Сумма ТРЗ по эпикам")
        highlight = JsCode("""
            function(params) {
                let v = params.value;
                if (v > 40) {return {'color':'white','backgroundColor':'#d9534f'};}
                if (v >= 35 && v <= 49) {return {'backgroundColor':'#f0ad4e'};}
                return {};
            }
        """)
        gb_tasks = GridOptionsBuilder.from_dataframe(task_sum)
        gb_tasks.configure_default_column(resizable=True, sortable=True)
        gb_tasks.configure_column('Итого', cellStyle=highlight)
        AgGrid(task_sum, gridOptions=gb_tasks.build(), height=320, update_mode=GridUpdateMode.NO_UPDATE)
except Exception as e:
    st.warning(f"Не удалось построить свод по задачам: {e}")

df_f = df_g.dropna(subset=['Дата начала', 'Дата окончания']).copy()

col_start, col_end = st.columns(2)
with col_start:
    period_start = st.date_input(
        "Период: начало",
        value=pd.to_datetime('15.09.2025').date()#pd.to_datetime(df_f['Дата начала'].min()).date()
    )
with col_end:
    period_end   = st.date_input(
        "Период: конец",
        value=pd.to_datetime('15.11.2025').date()#pd.to_datetime(df_f['Дата окончания'].max()).date()
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
