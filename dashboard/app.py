import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from google.oauth2.service_account import Credentials
import gspread
from pandas.tseries.offsets import BusinessDay as BDay

# Локальные импорты (твои модули)
from plot_graph import plot_gantt, ui_gantt_settings, ui_theme_picker
from utils import transform_backlog_to_summary, working_hours_between, transform_gantt

# =============================================================================
# 1. НАСТРОЙКА СТРАНИЦЫ И НАВИГАЦИИ
# =============================================================================
st.set_page_config(page_title="Дашборд PO", layout="wide", initial_sidebar_state="expanded")

# Создаем боковое меню для удобной навигации
st.sidebar.title("🧭 Навигация")
app_mode = st.sidebar.radio(
    "Выберите раздел:",
    ["⚙️ Настройки и Данные", "📊 План и Эпики", "👥 Ресурсы и Риски"]
)
st.sidebar.markdown("---")

# =============================================================================
# 2. ПОДКЛЮЧЕНИЕ И ЗАГРУЗКА ДАННЫХ (ОПТИМИЗИРОВАНО)
# =============================================================================
@st.cache_resource
def get_gspread_client():
    """Создать авторизованный клиент gspread из секретов"""
    creds = Credentials.from_service_account_info(
        st.secrets['google_service_account'], 
        scopes=st.secrets['excel']['SCOPES']
    )
    return gspread.authorize(creds)

# Кешируем данные на 10 минут (ttl=600), чтобы интерфейс не тормозил при кликах
@st.cache_data(ttl=600, show_spinner=False)
def load_all_data():
    gc = get_gspread_client()
    sh = gc.open_by_url(st.secrets['excel']['SHEET_URL'])
    
    return {
        'raw_df': pd.DataFrame(sh.worksheet('Декомпозиция (story, enablers)').get_all_records()),
        'df_people': pd.DataFrame(sh.worksheet('Штат(дашборд)').get_all_records()),
        'df_sprint': pd.DataFrame(sh.worksheet('Спринты(дашборд)').get_all_records()),
        'df_leave': pd.DataFrame(sh.worksheet('Отпуска').get_all_records()),
        'df_rates': pd.DataFrame(sh.worksheet('Ставка').get_all_records()),
        'df_holidays': pd.DataFrame(sh.worksheet('Праздники').get_all_records())
    }

with st.spinner("🔄 Загрузка данных из Google Таблиц..."):
    # Если нажата кнопка принудительного обновления (которую мы добавим дальше), кэш сбросится
    data = load_all_data()

# Распаковываем в удобные переменные для дальнейшего использования
df_raw = data['raw_df']
df_sprint = data['df_sprint']
df_people = data['df_people']
df_leave = data['df_leave']
df_rates = data['df_rates']
df_holidays = data['df_holidays']

# Кнопка принудительного обновления в сайдбаре
if st.sidebar.button("🔄 Обновить данные из БД", use_container_width=True):
    load_all_data.clear() # Очищаем кэш
    st.rerun()            # Перезапускаем приложение


# =============================================================================
# 3. РАЗДЕЛ 1: НАСТРОЙКИ И ДАННЫЕ
# =============================================================================
if app_mode == "⚙️ Настройки и Данные":
    st.title("⚙️ 1. Настройка и фильтрация данных")
    st.info("Выберите нужные колонки и параметры для фильтрации, затем нажмите 'Применить'.")

    # Ищем колонки
    col_quarter = 'Квартал 2026' if 'Квартал 2026' in df_raw.columns else None
    col_core = 'CORE' if 'CORE' in df_raw.columns else None

    with st.form("init_settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            epic_idx = 14 if len(df_raw.columns) > 14 else 0
            epic_col = st.selectbox("Колонка с Эпиками", df_raw.columns.tolist(), index=epic_idx)
            
            if col_quarter:
                q_vals = [v for v in df_raw[col_quarter].dropna().unique() if str(v).strip()]
                selected_q = st.selectbox(f"Квартал ({col_quarter})", ["Все"] + list(q_vals))
            else:
                selected_q = "Все"
                st.warning("Колонка с кварталом не найдена в таблице.")
                
        with col2:
            task_idx = 18 if len(df_raw.columns) > 18 else 0
            task_col = st.selectbox("Колонка с Задачами", df_raw.columns.tolist(), index=task_idx)
            
            if col_core:
                c_vals = [v for v in df_raw[col_core].dropna().unique() if str(v).strip()]
                selected_c = st.selectbox(f"CORE ({col_core})", ["Все"] + list(c_vals))
            else:
                selected_c = "Все"
                st.warning("Колонка CORE не найдена в таблице.")
        
        st.error("! Выберите режим отображения: Эпики или Задачи. В режиме Эпиков ТРЗ суммируется по всем задачам эпика.")
        mode = st.radio("Режим:", ["Эпики", "Задачи"], index=0, horizontal=True)

        apply_btn = st.form_submit_button("🚀 Применить", type="primary")

    # Сохраняем настройки
    if apply_btn:
        st.session_state.is_applied = True
        st.session_state.epic_col = epic_col
        st.session_state.task_col = task_col
        st.session_state.selected_q = selected_q
        st.session_state.selected_c = selected_c
        st.session_state.mode = mode
        st.success("Настройки применены! Можно редактировать таблицу.")

    # Если еще не применили настройки — останавливаем отрисовку страницы здесь
    if not st.session_state.get('is_applied', False):
        st.stop()

    # --- Применяем фильтры ---
    df_filtered = df_raw.copy()
    if col_quarter and st.session_state.selected_q != "Все":
        df_filtered = df_filtered[df_filtered[col_quarter] == st.session_state.selected_q]
        
    if col_core and st.session_state.selected_c != "Все":
        df_filtered = df_filtered[df_filtered[col_core] == st.session_state.selected_c]

    st.write("---")
    st.subheader("🔍 2. Фильтрация и редактирование данных (AgGrid)")
    st.info("Отфильтруйте нужные данные через меню колонок. После подготовки таблицы переходите к другим разделам в меню слева.")

    gb = GridOptionsBuilder.from_dataframe(df_filtered)
    gb.configure_default_column(editable=True, filter=True, sortable=True, resizable=True, minWidth=100)
    grid_options = gb.build()

    grid_response = AgGrid(
        df_filtered,
        gridOptions=grid_options,
        enable_enterprise_modules=False,
        update_mode=(GridUpdateMode.MODEL_CHANGED | GridUpdateMode.FILTERING_CHANGED),
        theme='streamlit',
        fit_columns_on_grid_load=False,
        height=500,
        key="main_source_grid"
    )

    # Сохраняем результат редактирования в session_state для других страниц
    if grid_response['data'] is not None and len(grid_response['data']) > 0:
        st.session_state.df = pd.DataFrame(grid_response['data'])

# =============================================================================
# ПРЕДОХРАНИТЕЛЬ ДЛЯ ОСТАЛЬНЫХ СТРАНИЦ
# =============================================================================
# Если мы ушли на другую страницу, но данные еще не готовы
elif not st.session_state.get('is_applied', False) or 'df' not in st.session_state:
    st.warning("⚠️ Сначала перейдите в раздел '⚙️ Настройки и Данные', выберите параметры и нажмите 'Применить'.")
    st.stop()

# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ДЛЯ СЛЕДУЮЩИХ БЛОКОВ
# Если код дошел сюда, значит мы на 2 или 3 странице и данные готовы
if app_mode in ["📊 План и Эпики", "👥 Ресурсы и Риски"]:
    df = st.session_state.df
    column_epic_name = st.session_state.epic_col
    column_task_name = st.session_state.task_col
    mode = st.session_state.mode
    roles_list = ['Эксперт RnD', 'Аналитик', 'Архитектор', 'UX/UI', 'C#', 'Py', 'React', 'QA', 'DevOps']


# =============================================================================
# ОБЩИЕ РАСЧЕТЫ ДЛЯ РАЗДЕЛОВ 2 И 3 (Подготовка df_g)
# =============================================================================
if app_mode in ["📊 План и Эпики", "👥 Ресурсы и Риски"]:
    # 1. Трансформация бэклога
    df_g = transform_backlog_to_summary(df, df_sprint, column_epic_name, column_task_name, mode)
    
    # 2. Подготовка дат и ТРЗ
    df_sprint["Номер спринта"] = pd.to_numeric(df_sprint["Номер спринта"], errors="coerce").astype("Int64")
    df_g["Номер спринта"] = pd.to_numeric(df_g["Номер спринта"], errors="coerce").astype("Int64")
    
    df_g = df_g.merge(
        df_sprint[['Номер спринта', 'Дата начала', 'Дата окончания']],
        on='Номер спринта', how='left'
    ).rename(columns={'Дата начала': 'Дата начала спринта', 'Дата окончания': 'Дата окончания спринта'})

    df_g['Sprint_Start'] = pd.to_datetime(df_g['Дата начала спринта'], format='%d.%m.%Y', errors='coerce')
    df_g['Sprint_End'] = pd.to_datetime(df_g['Дата окончания спринта'], format='%d.%m.%Y', dayfirst=True, errors='coerce')
    df_g['ТРЗ'] = pd.to_numeric(df_g['ТРЗ'], errors='coerce').fillna(0).round().astype(int)

    task_keys = [c for c in ['Название задачи', 'Номер спринта'] if c in df_g.columns] or ['Название задачи']
    
    df_g['Роль'] = df_g['Роль'].astype(str).str.strip()
    role_to_phase = {
        'Аналитик': 'A', 'Архитектор': 'A', 'Эксперт RnD': 'A', 'UX/UI': 'A', 'Системный аналитик': 'A',
        'QA': 'Q', 'Тестировщик': 'Q', 'Тестирование': 'Q'
    }
    df_g['phase'] = df_g['Роль'].map(role_to_phase).fillna('D')

    # 3. Расчет каскадной модели A -> D -> Q
    # (A) Аналитика
    mask_A = df_g['phase'].eq('A')
    df_g.loc[mask_A, 'Start_calc'] = df_g.loc[mask_A, 'Sprint_Start']
    df_g.loc[mask_A, 'Finish_calc'] = df_g.loc[mask_A, 'Start_calc'] + df_g.loc[mask_A, 'ТРЗ'].apply(BDay)
    a_finish = df_g[mask_A].groupby(task_keys, dropna=False)['Finish_calc'].max().reset_index().rename(columns={'Finish_calc': 'A_finish'})
    df_g = df_g.merge(a_finish, on=task_keys, how='left')
    df_g['A_finish'] = df_g['A_finish'].fillna(df_g['Sprint_Start'])

    # (D) Разработка
    mask_D = df_g['phase'].eq('D')
    df_g.loc[mask_D, 'Start_calc'] = df_g.loc[mask_D, 'A_finish']
    df_g.loc[mask_D, 'Finish_calc'] = df_g.loc[mask_D, 'Start_calc'] + df_g.loc[mask_D, 'ТРЗ'].apply(BDay)
    d_finish = df_g[mask_D].groupby(task_keys, dropna=False)['Finish_calc'].max().reset_index().rename(columns={'Finish_calc': 'D_finish'})
    df_g = df_g.merge(d_finish, on=task_keys, how='left')
    df_g['D_finish'] = df_g['D_finish'].fillna(df_g['A_finish'])

    # (Q) QA
    mask_Q = df_g['phase'].eq('Q')
    df_g.loc[mask_Q, 'Start_calc'] = df_g.loc[mask_Q, 'D_finish']
    df_g.loc[mask_Q, 'Finish_calc'] = df_g.loc[mask_Q, 'Start_calc'] + df_g.loc[mask_Q, 'ТРЗ'].apply(BDay)

    df_g['Start_calc'] = pd.to_datetime(df_g['Start_calc'])
    df_g['Finish_calc'] = pd.to_datetime(df_g['Finish_calc'])
    df_g['Дата начала'] = df_g['Start_calc']
    df_g['Дата окончания'] = df_g['Finish_calc']

# =============================================================================
# 4. РАЗДЕЛ 2: ПЛАН И ЭПИКИ
# =============================================================================
if app_mode == "📊 План и Эпики":
    
    # --- 2.1: Сводка по Эпикам ---
    st.header("📋 Сводка по Эпикам")
    unique_epics_count = df[column_epic_name].nunique()
    st.metric("Уникальных эпиков в работе", unique_epics_count)

    trz_cols_to_sum = [f"трз {role}" for role in roles_list if f"трз {role}" in df.columns]
    if trz_cols_to_sum:
        df_epic_calc = df.copy()
        for col in trz_cols_to_sum:
            df_epic_calc[col] = pd.to_numeric(df_epic_calc[col].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)
        
        epic_trz_summary = df_epic_calc.groupby(column_epic_name)[trz_cols_to_sum].sum()
        epic_trz_summary['Общий ТРЗ'] = epic_trz_summary.sum(axis=1)
        epic_trz_summary = epic_trz_summary.sort_values(by='Общий ТРЗ', ascending=False).reset_index()

        # Переставляем Общий ТРЗ вперед
        cols = epic_trz_summary.columns.tolist()
        cols.insert(1, cols.pop(cols.index('Общий ТРЗ')))
        epic_trz_summary = epic_trz_summary[cols]

        def highlight_high_trz(val):
            return 'background-color: #d9534f; color: white; font-weight: bold' if val > 40 else ''

        st.dataframe(
            epic_trz_summary.style.map(highlight_high_trz, subset=['Общий ТРЗ']),
            use_container_width=True, hide_index=True
        )

    st.write("---")

    # --- 2.2: Нагрузка по спринтам ---
    st.header("📅 Нагрузка по спринтам")
    df_sprints_chart = df_g.dropna(subset=['Номер спринта']).copy()

    if not df_sprints_chart.empty:
        # Фикс Plotly
        df_sprints_chart['Спринт_текст'] = "Спринт " + df_sprints_chart['Номер спринта'].astype(int).astype(str)
        sprint_stats = df_sprints_chart.groupby(['Спринт_текст', 'Роль'])['ТРЗ'].sum().reset_index()
        sprint_totals = sprint_stats.groupby('Спринт_текст')['ТРЗ'].sum().reset_index()
        
        sorted_sprints_nums = sorted(df_sprints_chart['Номер спринта'].astype(int).unique())
        category_order = [f"Спринт {num}" for num in sorted_sprints_nums]
        
        theme_key, theme, template_name = ui_theme_picker(expanded=False, default_key="pastel")

        fig_sprints = px.bar(
            sprint_stats, x='Спринт_текст', y='ТРЗ', color='Роль',
            title="Объем задач (ТРЗ) по спринтам",
            labels={'ТРЗ': 'Трудозатраты', 'Спринт_текст': 'Спринт'},
            template=template_name, text_auto='.1f'
        )

        fig_sprints.update_layout(
            xaxis={'type': 'category', 'categoryorder': 'array', 'categoryarray': category_order}, 
            barmode='stack', height=500
        )

        for _, row in sprint_totals.iterrows():
            fig_sprints.add_annotation(
                x=row['Спринт_текст'], y=row['ТРЗ'],
                text=f"<b>{row['ТРЗ']:.1f}</b>", showarrow=False, yshift=25,
                font=dict(color=theme.get("text_color", "white") if theme else "white")
            )

        st.plotly_chart(fig_sprints, use_container_width=True,theme=None)
        st.caption(f"Средняя нагрузка: **{sprint_totals['ТРЗ'].mean():.1f}** чел/дн.")

    st.write("---")

    # --- 2.3: Гант-диаграммы ---
    st.header("📈 Графики Ганта")
    tab1, tab2 = st.tabs(["Поэтапная (A->D->Q)", "Компактная (Суммарная)"])

    with tab1:
        st.info("Учитывает последовательность разработки: Аналитика -> Разработка -> QA.")
        settings_gant1 = ui_gantt_settings(df_g, prefix="gantt1", title="⚙️ Настройки", default_y_idx=0, default_task_idx=6, default_color_idx=0, default_sort_first="Resource")
        gantt_df = df_g[[settings_gant1['task_col'], settings_gant1['start_col'], settings_gant1['end_col'], 'Y_Group', settings_gant1['color_by']]].copy()
        gantt_df.columns = ["Task", "Start", "Finish", "Y_Group", "Resource"]
        asc_desc = True if settings_gant1['asc_desc'] == "ASC" else False

        plot_gantt(
            gantt_df.sort_values(by=[settings_gant1['sort_val_first'], "Start"], ascending=asc_desc),
            graph_title="График занятости", font_size=settings_gant1["font_size"],
            width_plot=settings_gant1["width_plot"], height_plot=settings_gant1["height_plot"],
            fit_names=settings_gant1['fit_names'], font_size_names=settings_gant1['font_size_names'],
            max_len=settings_gant1['max_len'], swap_text=settings_gant1['swap_text'],
            theme=theme, sprint_df=df_sprint
        )

    with tab2:
        st.write("Суммирует ТРЗ по всем ролям внутри эпика.")
        lane_df = transform_gantt(gantt_df)
        settings_gant2 = ui_gantt_settings(lane_df, prefix="gantt2", title="⚙️ Настройки", default_y_idx=5, default_task_idx=0, default_color_idx=4)
        
        plot_gantt(
            lane_df, start_column="Start", end_column="Finish", y_group="Track", color_column=settings_gant2["color_by"],
            text_column=settings_gant2["task_col"], graph_title="Компактный график", font_size=settings_gant2["font_size"],
            width_plot=settings_gant2["width_plot"], height_plot=settings_gant2["height_plot"],
            fit_names=settings_gant2['fit_names'], font_size_names=settings_gant2['font_size_names'],
            max_len=settings_gant2['max_len'], swap_text=settings_gant2['swap_text'],
            theme=theme, sprint_df=df_sprint
        )


# =============================================================================
# 5. РАЗДЕЛ 3: РЕСУРСЫ И РИСКИ
# =============================================================================
elif app_mode == "👥 Ресурсы и Риски":
    st.title("👥 Аналитика ресурсов и управление рисками")

    # --- 1. Настройка периода ---
    st.subheader("🗓️ Период анализа")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        analysis_start = st.date_input("Начало периода", value=pd.to_datetime('2026-04-06').date())
    with col_p2:
        analysis_end = st.date_input("Конец периода", value=pd.to_datetime('2026-06-26').date())

    if analysis_start > analysis_end:
        st.error("Ошибка: Дата начала не может быть позже даты окончания.")
        st.stop()

    st.write("---")

    # --- Подготовка общих справочников ---
    total_working_days = np.busday_count(analysis_start, analysis_end + pd.Timedelta(days=1))
    rates_map = dict(zip(df_rates['Исполнитель'], df_rates['Ставка']))
    df_holidays['Дата_dt'] = pd.to_datetime(df_holidays['Дата'], dayfirst=True, errors='coerce').dt.date
    general_holidays = df_holidays[df_holidays['Исполнитель'].str.lower() == 'все']['Дата_dt'].dropna().tolist()

    person_to_role = {}
    for role in roles_list:
        if role in df_people.columns:
            for n in df_people[role].dropna().unique():
                if str(n).strip() and str(n).lower() != 'nan':
                    person_to_role[str(n).strip()] = role

    # Функция для подсветки дефицита ТРЗ
    def style_negative(v):
        return 'color: #d9534f; font-weight: bold' if isinstance(v, (int, float)) and v < 0 else None

    # =========================================================
    # БЛОК 3.1: Сводная емкость по стекам (Capacity)
    # =========================================================
    st.subheader("📊 Баланс ресурсов по стекам")
    
    available_dict, occupied_dict = {}, {}
    for role in roles_list:
        role_capacity = 0
        if role in df_people.columns:
            staff_list = [name for name in df_people[role].unique() if str(name).strip() and str(name).lower() != 'nan']
            for person in staff_list:
                try: fte = float(str(rates_map.get(person, 1.0)).replace(',', '.'))
                except: fte = 1.0
                
                person_holidays = df_holidays[df_holidays['Исполнитель'] == person]['Дата_dt'].dropna().tolist()
                combined_holidays = list(set(general_holidays + person_holidays))
                
                wd_count = np.busday_count(analysis_start, analysis_end + pd.Timedelta(days=1), holidays=combined_holidays)
                
                person_vacation_days = 0
                person_vacations = df_leave[df_leave['Исполнитель'] == person]
                for _, vac in person_vacations.iterrows():
                    v_start = pd.to_datetime(vac['НАЧАЛО'], dayfirst=True, errors='coerce').date()
                    v_end = pd.to_datetime(vac['КОНЕЦ'], dayfirst=True, errors='coerce').date()
                    if pd.isna(v_start) or pd.isna(v_end): continue
                    
                    overlap_start, overlap_end = max(v_start, analysis_start), min(v_end, analysis_end)
                    if overlap_start <= overlap_end:
                        person_vacation_days += np.busday_count(overlap_start, overlap_end + pd.Timedelta(days=1), holidays=combined_holidays)
                
                role_capacity += (wd_count - person_vacation_days) * fte
                
        available_dict[role] = round(role_capacity, 1)

        trz_col = f"трз {role}"
        if trz_col in df.columns:
            val = pd.to_numeric(df[trz_col].astype(str).str.replace(',', '.'), errors='coerce').fillna(0).sum()
            occupied_dict[role] = round(val, 1)
        else:
            occupied_dict[role] = 0

    summary_records = [{"Стек": r, "Доступно (чел/дн)": available_dict[r], "Занято (ТРЗ)": occupied_dict[r], "Свободно": round(available_dict[r] - occupied_dict[r], 1)} for r in roles_list]
    df_final_stats = pd.DataFrame(summary_records).set_index("Стек").T
    
    st.dataframe(df_final_stats.style.map(style_negative, subset=pd.IndexSlice["Свободно", :]), use_container_width=True)

    # =========================================================
    # БЛОК 3.2: Индивидуальная загрузка
    # =========================================================
    st.subheader("👤 Индивидуальная загрузка")
    person_occupied = df_g.groupby('Исполнитель')['ТРЗ'].sum().to_dict()
    individual_stats = []

    for role in roles_list:
        if role in df_people.columns:
            staff_list = [name for name in df_people[role].unique() if str(name).strip() and str(name).lower() != 'nan']
            for person in staff_list:
                try: fte = float(str(rates_map.get(person, 1.0)).replace(',', '.'))
                except: fte = 1.0
                
                person_holidays = df_holidays[df_holidays['Исполнитель'] == person]['Дата_dt'].dropna().tolist()
                combined_holidays = list(set(general_holidays + person_holidays))
                wd_count = np.busday_count(analysis_start, analysis_end + pd.Timedelta(days=1), holidays=combined_holidays)
                
                person_vac_days = 0
                for _, vac in df_leave[df_leave['Исполнитель'] == person].iterrows():
                    v_start = pd.to_datetime(vac['НАЧАЛО'], dayfirst=True, errors='coerce').date()
                    v_end = pd.to_datetime(vac['КОНЕЦ'], dayfirst=True, errors='coerce').date()
                    if not (pd.isna(v_start) or pd.isna(v_end)) and max(v_start, analysis_start) <= min(v_end, analysis_end):
                        person_vac_days += np.busday_count(max(v_start, analysis_start), min(v_end, analysis_end) + pd.Timedelta(days=1), holidays=combined_holidays)
                
                cap = (wd_count - person_vac_days) * fte
                occ = person_occupied.get(person, 0.0)
                individual_stats.append({'Сотрудник': person, 'Стек': role, 'Ставка': fte, 'Дней отпуска': person_vac_days, 'Доступно ТРЗ': round(cap, 1), 'Занято ТРЗ': round(occ, 1), 'Свободно ТРЗ': round(cap - occ, 1)})

    if individual_stats:
        df_indiv = pd.DataFrame(individual_stats).sort_values(by=['Стек', 'Сотрудник'])
        with st.expander("Показать таблицу детальной загрузки", expanded=False):
            st.dataframe(df_indiv.style.map(style_negative, subset=['Свободно ТРЗ']), use_container_width=True, hide_index=True)

    # =========================================================
    # БЛОК 3.3: Календарь отсутствий
    # =========================================================
    st.subheader("🏖️ Календарь отсутствий и сетка спринтов")
    calendar_data = []
    
    # Отпуска
    for _, row in df_leave.iterrows():
        person, v_start, v_end = row['Исполнитель'], pd.to_datetime(row['НАЧАЛО'], dayfirst=True, errors='coerce').date(), pd.to_datetime(row['КОНЕЦ'], dayfirst=True, errors='coerce').date()
        if not (pd.isna(v_start) or pd.isna(v_end)) and v_start <= analysis_end and v_end >= analysis_start:
            role = person_to_role.get(person, "Вне штата")
            calendar_data.append(dict(Сотрудник=person, Начало=max(v_start, analysis_start), Конец=min(v_end, analysis_end) + pd.Timedelta(days=1), Тип='Отпуск', Группа_цвета=role, Детали=f"Отпуск ({role}): {v_start.strftime('%d.%m')} - {v_end.strftime('%d.%m')}"))

    # Праздники (общие и личные)
    all_staff = set(person_to_role.keys())
    for _, row in df_holidays.iterrows():
        h_date = row['Дата_dt']
        targets_raw = str(row['Исполнитель']).strip()
        
        if not pd.isna(h_date) and analysis_start <= h_date <= analysis_end:
            # 1. Если праздник для всех
            if targets_raw.lower() == 'все':
                for p in all_staff:
                    calendar_data.append(dict(
                        Сотрудник=p, 
                        Начало=h_date, 
                        Конец=h_date + pd.Timedelta(days=1), 
                        Тип='Праздник', 
                        Группа_цвета='Праздник (Общий)', 
                        Детали=f"Общий праздник: {h_date.strftime('%d.%m')}"
                    ))
            # 2. Если указаны конкретные люди (поддержка списка через запятую)
            else:
                # Разделяем строку по запятой и убираем лишние пробелы у каждого имени
                target_list = [t.strip() for t in targets_raw.split(',') if t.strip()]
                for target in target_list:
                    calendar_data.append(dict(
                        Сотрудник=target, 
                        Начало=h_date, 
                        Конец=h_date + pd.Timedelta(days=1), 
                        Тип='Праздник', 
                        Группа_цвета='Праздник (Личный)', 
                        Детали=f"Личный праздник: {h_date.strftime('%d.%m')}"
                    ))

    if calendar_data:
        df_cal = pd.DataFrame(calendar_data)
        unique_people = list(df_cal['Сотрудник'].unique())
        unique_people.sort(key=lambda p: (person_to_role.get(p, "ЯЯ_Вне штата"), p))
        sorted_names = list(reversed(unique_people))

        fig_cal = px.timeline(df_cal, x_start="Начало", x_end="Конец", y="Сотрудник", color="Группа_цвета", hover_name="Детали", color_discrete_map={'Праздник (Общий)': '#d62728', 'Праздник (Личный)': '#ff7f0e'}, template="plotly_white")
        fig_cal.update_traces(width=0.8) 

        fig_cal.update_layout(
            barmode='overlay', 
            height=max(400, len(sorted_names) * 30), 
            legend_title_text="Категория"
        )
        # if not df_sprint.empty:
        #     for _, spr in df_sprint.iterrows():
        #         s_start, s_end, s_num = pd.to_datetime(spr['Дата начала'], dayfirst=True, errors='coerce').date(), pd.to_datetime(spr['Дата окончания'], dayfirst=True, errors='coerce').date(), spr.get('Номер спринта', '?')
        #         if not (pd.isna(s_start) or pd.isna(s_end)) and s_start <= analysis_end and s_end >= analysis_start:
        #             fig_cal.add_vrect(x0=s_start, x1=s_end, fillcolor="rgba(100, 150, 255, 0.08)" if int(s_num or 0) % 2 == 0 else "rgba(100, 150, 255, 0.02)", line_width=0, layer="below")
        #             fig_cal.add_annotation(x=s_start + (s_end - s_start)/2, y=1.02, yref="paper", text=f"Спринт {s_num}", showarrow=False, font=dict(size=10, color="gray"), textangle=0 if (s_end - s_start).days > 7 else -90)
        # --- СЕТКА СПРИНТОВ (с непрерывными границами) ---
        if not df_sprint.empty:
            # Создаем копию и сортируем спринты по дате начала
            df_sprint_sorted = df_sprint.copy()
            df_sprint_sorted['dt_start'] = pd.to_datetime(df_sprint_sorted['Дата начала'], dayfirst=True, errors='coerce')
            df_sprint_sorted['dt_end'] = pd.to_datetime(df_sprint_sorted['Дата окончания'], dayfirst=True, errors='coerce')
            df_sprint_sorted = df_sprint_sorted.dropna(subset=['dt_start']).sort_values('dt_start')
            
            sprints_records = df_sprint_sorted.to_dict('records')
            
            for i, spr in enumerate(sprints_records):
                s_start = spr['dt_start'].date()
                s_num = spr.get('Номер спринта', '?')
                
                if pd.isna(spr['dt_end']): 
                    continue
                
                # Базовая дата окончания
                s_end = spr['dt_end'].date()
                
                # Магия для красоты: если есть следующий спринт, продлеваем границу текущего до его начала
                if i + 1 < len(sprints_records):
                    next_start = sprints_records[i+1]['dt_start'].date()
                    if next_start > s_end:
                        s_end = next_start
                
                # Отрисовка зоны, если она попадает в выбранный период
                if s_start <= analysis_end and s_end >= analysis_start:
                    fill_color = "rgba(100, 150, 255, 0.08)" if int(s_num or 0) % 2 == 0 else "rgba(100, 150, 255, 0.02)"
                    
                    fig_cal.add_vrect(
                        x0=s_start, x1=s_end, 
                        fillcolor=fill_color, line_width=0, layer="below"
                    )
                    
                    fig_cal.add_annotation(
                        x=s_start + (s_end - s_start)/2, 
                        y=1.02, yref="paper", 
                        text=f"Спринт {s_num}", showarrow=False, 
                        font=dict(size=10, color="gray"), 
                        textangle=0 if (s_end - s_start).days > 7 else -90
                    )

        fig_cal.update_yaxes(categoryorder="array", categoryarray=sorted_names, title="")
        fig_cal.update_xaxes(tickformat="%d.%m", range=[analysis_start, analysis_end])
        fig_cal.update_layout(barmode='relative',height=max(400, len(sorted_names) * 30), legend_title_text="Категория")
        st.plotly_chart(fig_cal, use_container_width=True,theme=None)

    # =========================================================
    # БЛОК 3.4: Риски (Пересечения отпусков и конфликты с задачами)
    # =========================================================
    col_risk1, col_risk2 = st.columns(2)
    
    with col_risk1:
        st.subheader("⚠️ Пересечения отпусков в стеке")
        df_vac = pd.DataFrame([r for r in calendar_data if r['Тип'] == 'Отпуск'])
        overlaps = []
        if not df_vac.empty:
            for role, group in df_vac.groupby('Группа_цвета'):
                records = group.to_dict('records')
                for i in range(len(records)):
                    for j in range(i + 1, len(records)):
                        v1, v2 = records[i], records[j]
                        if v1['Сотрудник'] != v2['Сотрудник'] and v1['Начало'] <= v2['Конец'] and v2['Начало'] <= v1['Конец']:
                            o_start, o_end = max(v1['Начало'], v2['Начало']), min(v1['Конец'], v2['Конец']) - pd.Timedelta(days=1)
                            overlap_days = np.busday_count(o_start, o_end + pd.Timedelta(days=1), holidays=general_holidays)
                            if overlap_days > 0:
                                overlaps.append({'Стек': role, 'Сотрудник 1': v1['Сотрудник'], 'Сотрудник 2': v2['Сотрудник'], 'Дни пересечения': f"{o_start.strftime('%d.%m')} - {o_end.strftime('%d.%m')}", 'Раб. дней риска': overlap_days})

        if overlaps:
            df_overlaps = pd.DataFrame(overlaps).sort_values(by=['Стек', 'Раб. дней риска'], ascending=[True, False])
            st.dataframe(df_overlaps.style.map(lambda v: 'color: #d9534f; font-weight: bold' if v > 3 else '', subset=['Раб. дней риска']), use_container_width=True, hide_index=True)
        else:
            st.success("Критичных пересечений нет.")

    with col_risk2:
        st.subheader("🚩 Конфликты: Задачи vs Отпуск")
        task_conflicts = []
        for _, task in df_g.iterrows():
            person = task['Исполнитель']
            if not person or str(person).lower() == 'nan': continue
            t_start = task['Дата начала'].date() if isinstance(task['Дата начала'], pd.Timestamp) else task['Дата начала']
            t_end = task['Дата окончания'].date() if isinstance(task['Дата окончания'], pd.Timestamp) else task['Дата окончания']
            
            for _, vac in df_leave[df_leave['Исполнитель'] == person].iterrows():
                v_start, v_end = pd.to_datetime(vac['НАЧАЛО'], dayfirst=True, errors='coerce').date(), pd.to_datetime(vac['КОНЕЦ'], dayfirst=True, errors='coerce').date()
                if not (pd.isna(v_start) or pd.isna(v_end)) and t_start <= v_end and v_start <= t_end:
                    c_start, c_end = max(t_start, v_start), min(t_end, v_end)
                    combined_h = list(set(general_holidays + df_holidays[df_holidays['Исполнитель'] == person]['Дата_dt'].dropna().tolist()))
                    conflict_days = np.busday_count(c_start, c_end + pd.Timedelta(days=1), holidays=combined_h)
                    
                    if conflict_days > 0:
                        task_conflicts.append({'Стек': task['Роль'], 'Исполнитель': person, 'Эпик': task.get('Название крупная задача (ЭПИК)', ''), 'Задача': task['Название задачи'], 'Дней конфликта': conflict_days})

        if task_conflicts:
            df_task_conf = pd.DataFrame(task_conflicts).sort_values(by='Дней конфликта', ascending=False)
            st.dataframe(df_task_conf.style.map(lambda v: 'color: #d9534f; font-weight: bold' if v > 2 else '', subset=['Дней конфликта']), use_container_width=True, hide_index=True)
        else:
            st.success("Задачи обеспечены исполнителями.")