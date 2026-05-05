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
                text=f"<b>{row['ТРЗ']:.1f}</b>", showarrow=False, yshift=10,
                font=dict(color=theme.get("text_color", "white") if theme else "white")
            )

        st.plotly_chart(fig_sprints, use_container_width=True)
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
            theme=theme, sprint_df=st.session_state.df_sprint
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
            theme=theme, sprint_df=st.session_state.df_sprint
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
        fig_cal.update_layout(height=max(400, len(sorted_names) * 30), legend_title_text="Категория")
        st.plotly_chart(fig_cal, use_container_width=True)

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




# import streamlit as st
# from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
# from st_aggrid.shared import JsCode


# from google.oauth2.service_account import Credentials
# from gspread_dataframe import set_with_dataframe, get_as_dataframe
# # from gspread.worksheet import Worksheet
# import gspread
# import pandas as pd
# # from gspread.worksheet import Worksheet

# import pandas as pd
# from pandas.tseries.offsets import BusinessDay as BDay
# # from functools import partial
# import numpy as np
# import plotly.express as px
# from plot_graph import plot_gantt, ui_gantt_settings,ui_theme_picker
# from utils import transform_backlog_to_summary, \
#                 working_hours_between,\
#                 transform_gantt
#                 # create_pivot,\
#                 # wrap_text,\
#                 # assign_tracks,\
                

# ##############################################################################
# # ---  Настройка страницы --- 
# st.set_page_config(
#     page_title="Мой дашборд", layout="wide", initial_sidebar_state="expanded")

# ##############################################################################


# SHEET_URL = st.secrets['excel']['SHEET_URL']
# SERVICE_ACCOUNT = st.secrets['google_service_account']
# SCOPES = st.secrets['excel']['SCOPES']
# def get_gspread_client():
#     """Создать авторизованный клиент gspread из секретов"""
#     creds = Credentials.from_service_account_info(SERVICE_ACCOUNT, scopes=SCOPES)
#     return gspread.authorize(creds)
# gc = get_gspread_client()

# sh = gc.open_by_url(SHEET_URL)

# ##############################################################################
# # --- Основные настройки --- 
# worksheet_1 = sh.worksheet('Декомпозиция (story, enablers)')
# worksheet_2 = sh.worksheet('Штат(дашборд)')
# worksheet_3 = sh.worksheet('Спринты(дашборд)')
# worksheet_4 = sh.worksheet('Отпуска')
# worksheet_rates = sh.worksheet('Ставка')
# worksheet_holidays = sh.worksheet('Праздники')


# # Сохраняем сырые данные в session_state, чтобы не дергать API Google при каждом реране
# if 'raw_df' not in st.session_state:
#     st.session_state.raw_df     = pd.DataFrame(worksheet_1.get_all_records())
#     st.session_state.df_people  = pd.DataFrame(worksheet_2.get_all_records())
#     st.session_state.df_sprint  = pd.DataFrame(worksheet_3.get_all_records())
#     st.session_state.df_leave   = pd.DataFrame(worksheet_4.get_all_records())
#     st.session_state.df_rates = pd.DataFrame(worksheet_rates.get_all_records())
#     st.session_state.df_holidays = pd.DataFrame(worksheet_holidays.get_all_records())

# df_raw = st.session_state.raw_df
# df_sprint = st.session_state.df_sprint
# df_people = st.session_state.df_people
# df_leave = st.session_state.df_leave
# df_rates = st.session_state.df_rates
# df_holidays = st.session_state.df_holidays

# ##############################################################################
# # --- БЛОК 1: Выбор исходных данных ---
# st.subheader("⚙️ 1. Настройка и фильтрация данных")
# st.info("Выберите нужные колонки и параметры для фильтрации, затем нажмите 'Применить'")

# # Ищем колонки Квартала и CORE (названия взяты из твоих закомментированных строк)
# col_quarter = 'Квартал 2026' if 'Квартал 2026' in df_raw.columns else None
# col_core = 'CORE' if 'CORE' in df_raw.columns else None

# with st.form("init_settings"):
#     col1, col2 = st.columns(2)
    
#     with col1:
#         epic_col = st.selectbox("Колонка с Эпиками", df_raw.columns.tolist(), index=14 if len(df_raw.columns) > 14 else 0)
        
#         # Динамически собираем уникальные значения для квартала, если колонка существует
#         if col_quarter:
#             q_vals = [v for v in df_raw[col_quarter].dropna().unique() if str(v).strip()]
#             selected_q = st.selectbox(f"Квартал ({col_quarter})", ["Все"] + list(q_vals))
#         else:
#             selected_q = "Все"
#             st.warning("Колонка с кварталом не найдена в таблице.")
            
#     with col2:
#         task_col = st.selectbox("Колонка с Задачами", df_raw.columns.tolist(), index=18 if len(df_raw.columns) > 18 else 0)
        
#         # Динамически собираем уникальные значения для CORE
#         if col_core:
#             c_vals = [v for v in df_raw[col_core].dropna().unique() if str(v).strip()]
#             selected_c = st.selectbox(f"CORE ({col_core})", ["Все"] + list(c_vals))
#         else:
#             selected_c = "Все"
#             st.warning("Колонка CORE не найдена в таблице.")
    
  
#     st.error("! Выберете режим отображения эпиков или задач! В режиме эпиков будут отображаться названия эпиков, при этом ТРЗ будут суммировать по всем задачам входящих в Эпик. В режиме Задача есть возможность анализировать отдельно задачи, входящие в эпик")
#     mode = st.radio(
#         "Выберите режим:",
#         ["Эпики", "Задачи"],
#         index=0,   # что выбрано по умолчанию
#         horizontal=True # в одну строку
#     )

#     apply_btn = st.form_submit_button("🚀 Применить", type="primary")



# # Сохраняем состояние при нажатии кнопки
# if apply_btn:
#     with st.spinner("🔄 Обновление данных из Google Таблиц..."):
#         # Принудительно запрашиваем свежие данные по всем листам
#         st.session_state.raw_df     = pd.DataFrame(worksheet_1.get_all_records())
#         st.session_state.df_people  = pd.DataFrame(worksheet_2.get_all_records())
#         st.session_state.df_sprint  = pd.DataFrame(worksheet_3.get_all_records())
#         st.session_state.df_leave   = pd.DataFrame(worksheet_4.get_all_records())
#         st.session_state.df_rates = pd.DataFrame(worksheet_rates.get_all_records())
#         st.session_state.df_holidays = pd.DataFrame(worksheet_holidays.get_all_records())
        
#     # Сохраняем выбранные настройки
#     st.session_state.is_applied = True
#     st.session_state.epic_col = epic_col
#     st.session_state.task_col = task_col
#     st.session_state.selected_q = selected_q
#     st.session_state.selected_c = selected_c
    
#     # Перезапускаем страницу, чтобы свежие данные подхватились в начале скрипта
#     st.rerun() 


# # МАГИЯ ЗДЕСЬ: Прерываем выполнение скрипта, если кнопка еще не была нажата
# if not st.session_state.get('is_applied', False):
#     st.stop()

# # --- Применяем фильтры ---
# df_filtered = df_raw.copy()
# if col_quarter and st.session_state.selected_q != "Все":
#     df_filtered = df_filtered[df_filtered[col_quarter] == st.session_state.selected_q]
    
# if col_core and st.session_state.selected_c != "Все":
#     df_filtered = df_filtered[df_filtered[col_core] == st.session_state.selected_c]

# # Передаем отфильтрованный датафрейм и названия колонок дальше по коду
# df = df_filtered
# column_epic_name = st.session_state.epic_col
# column_task_name = st.session_state.task_col

# ##############################################################################
# st.subheader("🔍 2. Фильтрация и редактирование данных")
# st.info('! Отфильтруйте нужные данные через меню в заголовках колонок. Как только таблица будет готова, нажмите кнопку "Построить графики" под ней.')

# # Настройка безопасного AgGrid
# gb = GridOptionsBuilder.from_dataframe(df)

# # Настраиваем все колонки разом: включаем базовые бесплатные фильтры
# gb.configure_default_column(
#     editable=True,
#     filter=True,       # Стандартный фильтр (Text/Number), который не крашит UI
#     sortable=True, 
#     resizable=True,
#     minWidth=100       # Безопасная ширина
# )

# grid_options = gb.build()

# # Отрисовка таблицы
# grid_response = AgGrid(
#     df,
#     gridOptions=grid_options,
#     enable_enterprise_modules=False,  # СТРОГО False, чтобы избежать белого экрана
#     update_mode=(
#         GridUpdateMode.MODEL_CHANGED | GridUpdateMode.FILTERING_CHANGED
#     ),
#     theme='streamlit',                # Стабильная нативная тема
#     fit_columns_on_grid_load=False,
#     height=450,
#     key="main_source_grid"
# )


# # Забираем отфильтрованные и отредактированные данные из AgGrid
# if grid_response['data'] is not None and len(grid_response['data']) > 0:
#     st.session_state.df = pd.DataFrame(grid_response['data'])
    
# df = st.session_state.df

# st.write('  ')
# st.write('  ')


# ##############################################################################
# # --- Раздел 2: Подготовка Gantt ---
# # st.subheader("Таблица для Ганта")
# st.session_state.df_gantt = transform_backlog_to_summary(df,df_sprint,column_epic_name,column_task_name,mode)
# df_g = st.session_state.df_gantt
# # --- merge со спринтами ---
# df_sprint["Номер спринта"] = pd.to_numeric(df_sprint["Номер спринта"], errors="coerce").astype("Int64")  # nullable int
# df_g["Номер спринта"] = pd.to_numeric(df_g["Номер спринта"], errors="coerce").astype("Int64")  # nullable int



# df_g = (df_g#st.session_state.df_gantt
#         .merge(df_sprint[['Номер спринта','Дата начала','Дата окончания']],
#                on='Номер спринта', how='left')
#         .rename(columns={'Дата начала':'Дата начала спринта',
#                          'Дата окончания':'Дата окончания спринта'}))

# # df_g['Sprint_Start'] = pd.to_datetime(df_g['Дата начала спринта'])
# # df_g['Sprint_End']   = pd.to_datetime(df_g['Дата окончания спринта'])
# df_g['Sprint_Start'] = pd.to_datetime(
#     df_g['Дата начала спринта'],
#     format='%d.%m.%Y',
#     # dayfirst=True,
#     # errors='coerce'
# )
# df_g['Sprint_End'] = pd.to_datetime(
#     df_g['Дата окончания спринта'],
#     format='%d.%m.%Y',
#     dayfirst=True,
#     # errors='coerce'
# )


# # TRZ -> int >= 1
# df_g['ТРЗ'] = (pd.to_numeric(df_g['ТРЗ'], errors='coerce')
#                  .fillna(0).round().astype(int))#.clip(lower=1))

# # --- ключи связки работ в одну «задачу» ---
# task_keys = [c for c in ['Название задачи','Номер спринта'] if c in df_g.columns]
# if not task_keys:  # запасной вариант
#     task_keys = ['Название задачи']
# # task_keys = [c for c in task_keys_all if c in df_g.columns]
# # df_g[task_keys] = df_g[task_keys].fillna('—')

# # --- фазовая модель: A (аналитика) -> D (dev) -> Q (QA) ---
# df_g['Роль'] = df_g['Роль'].astype(str).str.strip()

# role_to_phase = {
#     # Аналитика / подготовка
#     'Аналитик':'A', 'Архитектор':'A', 'Эксперт RnD':'A', 'UX/UI':'A', 'Системный аналитик':'A',
#     # QA
#     'QA':'Q', 'Тестировщик':'Q', 'Тестирование':'Q'
# }
# # всё, что не A и не Q, считаем разработкой (D)
# df_g['phase'] = df_g['Роль'].map(role_to_phase).fillna('D')

# # --- 1) Считаем A (аналитика): стартуют от начала спринта ---
# mask_A = df_g['phase'].eq('A')
# df_g.loc[mask_A, 'Start_calc']  = df_g.loc[mask_A, 'Sprint_Start']
# df_g.loc[mask_A, 'Finish_calc'] = df_g.loc[mask_A, 'Start_calc'] + df_g.loc[mask_A, 'ТРЗ'].apply(BDay)

# # Групповой финиш аналитики
# a_finish = (df_g[mask_A]
#             .groupby(task_keys, dropna=False)['Finish_calc']
#             .max()
#             .reset_index()
#             .rename(columns={'Finish_calc':'A_finish'}))
# df_g = df_g.merge(a_finish, on=task_keys, how='left')
# # если аналитики нет — берём начало спринта
# df_g['A_finish'] = df_g['A_finish'].fillna(df_g['Sprint_Start'])

# # --- 2) Считаем D (разработка): стартуют после A_finish ---
# mask_D = df_g['phase'].eq('D')
# df_g.loc[mask_D, 'Start_calc']  = df_g.loc[mask_D, 'A_finish']
# df_g.loc[mask_D, 'Finish_calc'] = df_g.loc[mask_D, 'Start_calc'] + df_g.loc[mask_D, 'ТРЗ'].apply(BDay)

# # Групповой финиш разработки
# d_finish = (df_g[mask_D]
#             .groupby(task_keys, dropna=False)['Finish_calc']
#             .max()
#             .reset_index()
#             .rename(columns={'Finish_calc':'D_finish'}))
# df_g = df_g.merge(d_finish, on=task_keys, how='left')
# # если разработки нет — финиш разработки = финишу аналитики
# df_g['D_finish'] = df_g['D_finish'].fillna(df_g['A_finish'])

# # --- 3) Считаем Q (QA): стартуют после D_finish (никогда раньше разработки) ---
# mask_Q = df_g['phase'].eq('Q')
# df_g.loc[mask_Q, 'Start_calc']  = df_g.loc[mask_Q, 'D_finish']
# df_g.loc[mask_Q, 'Finish_calc'] = df_g.loc[mask_Q, 'Start_calc'] + df_g.loc[mask_Q, 'ТРЗ'].apply(BDay)

# # Приводим типы
# df_g['Start_calc']  = pd.to_datetime(df_g['Start_calc'])
# df_g['Finish_calc'] = pd.to_datetime(df_g['Finish_calc'])

# # 6) Переименуем в понятные столбцы и отформатируем для AgGrid:
# df_g['Дата начала']   = df_g['Start_calc']#.dt.strftime('%Y-%m-%d')
# df_g['Дата окончания'] = df_g['Finish_calc']#.dt.strftime('%Y-%m-%d')



# #############################################################################
# # --- Настройка и отображение Gantt-диаграммы --- 
# st.subheader("Gantt-диаграмма поэтапная")
# st.info("Данная диаграмма учитывает последовательность разработки - сначала Аналитик/UX/UI/Архитектор (по макс времени), потом роли разработки (также по макс) и затем QA.")
# settings_gant1 = ui_gantt_settings(df_g, prefix="plotly", 
#                                    title="⚙️ Настройки",
#                                    default_y_idx=0,
#                                    default_task_idx=6,
#                                    default_color_idx=0,
#                                    default_sort_first="Resource")
# # подготовка данных
# # gantt_df = df_g[[settings_gant1['task_col'], settings_gant1['start_col'], settings_gant1['end_col'], settings_gant1['y_axis'], settings_gant1['color_by']]].copy()
# gantt_df = df_g[[settings_gant1['task_col'], settings_gant1['start_col'], settings_gant1['end_col'],'Y_Group', settings_gant1['color_by']]].copy()

# gantt_df[settings_gant1['start_col']] = pd.to_datetime(gantt_df[settings_gant1['start_col']], errors='coerce')
# gantt_df[settings_gant1['end_col']]   = pd.to_datetime(gantt_df[settings_gant1['end_col']], errors='coerce')
# gantt_df.columns = ["Task", "Start", "Finish", "Y_Group", "Resource"]

# if settings_gant1['asc_desc'] == "ASC":
#     asc_desc=True
# else:
#     asc_desc=False  
# theme_key, theme, template_name = ui_theme_picker(expanded=False, default_key="pastel",)
# with st.expander("График", expanded=True):
#     plot_gantt(gantt_df.sort_values(by=[settings_gant1['sort_val_first'],"Start"],ascending=asc_desc),
#                 graph_title="График занятости",
#                 font_size=settings_gant1["font_size"],
#                 width_plot=settings_gant1["width_plot"],
#                 height_plot =settings_gant1["height_plot"],
#                 fit_names=settings_gant1['fit_names'],
#                 font_size_names=settings_gant1['font_size_names'],
#                 max_len=settings_gant1['max_len'],
#                 swap_text=settings_gant1['swap_text'],
#                 theme=theme,
#                 sprint_df=st.session_state.df_sprint,
#                 )

# ##########################################################
# st.subheader("Gantt-диаграмма компактная")
# st.write("Данный график не учитывает последовательность разработки - по эпику суммируется ТРЗ по всем ролям.")
# lane_df = transform_gantt(gantt_df)
# settings_gant2 = ui_gantt_settings(lane_df, prefix="plotly1", 
#                                    title="⚙️ Настройки",
#                                    default_y_idx = 5, 
#                                    default_task_idx = 0,
#                                    default_color_idx = 4)
# theme_key1, theme1, template_name1 = ui_theme_picker(expanded=False, default_key="pastel",suffix="1")
# with st.expander("График", expanded=False):
#     plot_gantt(lane_df,
#                 start_column="Start",
#                 end_column="Finish",
#                 y_group="Track",#settings_gant2['y_axis'],
#                 color_column=settings_gant2["color_by"],
#                 text_column=settings_gant2["task_col"],
#                 graph_title="График занятости",
#                 font_size=settings_gant2["font_size"],
#                 width_plot=settings_gant2["width_plot"],
#                 height_plot =settings_gant2["height_plot"],
#                 fit_names=settings_gant2['fit_names'],
#                 font_size_names=settings_gant2['font_size_names'],
#                 max_len=settings_gant2['max_len'],
#                 swap_text=settings_gant2['swap_text'],
#                 theme=theme1,
#                 sprint_df=st.session_state.df_sprint,
#                 )



# ##############################################################################
# # --- Раздел 2.1: Сводка по Эпикам ---
# st.header("📋 2.1. Сводка по Эпикам")
# roles_list = ['Эксперт RnD', 'Аналитик', 'Архитектор', 'UX/UI', 'C#', 'Py', 'React', 'QA','DevOps']


# # 1. Считаем количество уникальных эпиков
# unique_epics_count = df[column_epic_name].nunique()
# st.metric("Количество уникальных эпиков в работе", unique_epics_count)

# # 2. Подготовка таблицы с суммарным ТРЗ по каждому эпику
# trz_cols_to_sum = [f"трз {role}" for role in roles_list if f"трз {role}" in df.columns]

# if trz_cols_to_sum:
#     # Создаем копию для расчетов
#     df_epic_calc = df.copy()
    
#     # Приводим все колонки ТРЗ к числовому виду
#     for col in trz_cols_to_sum:
#         df_epic_calc[col] = pd.to_numeric(df_epic_calc[col].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)
    
#     # Группируем по Эпику и суммируем все ТРЗ
#     epic_trz_summary = df_epic_calc.groupby(column_epic_name)[trz_cols_to_sum].sum()
    
#     # Считаем итоговый ТРЗ эпика
#     epic_trz_summary['Общий ТРЗ'] = epic_trz_summary.sum(axis=1)
    
#     # Сортируем по убыванию нагрузки и сбрасываем индекс
#     epic_trz_summary = epic_trz_summary.sort_values(by='Общий ТРЗ', ascending=False).reset_index()

#     # --- МЕНЯЕМ ПОРЯДОК КОЛОНОК ---
#     # Переставляем "Общий ТРЗ" с конца на второе место (индекс 1)
#     cols = epic_trz_summary.columns.tolist()
#     # Извлекаем 'Общий ТРЗ' и вставляем его по индексу 1
#     cols.insert(1, cols.pop(cols.index('Общий ТРЗ')))
#     epic_trz_summary = epic_trz_summary[cols]

#     st.subheader("Суммарные трудозатраты по Эпикам")
#     st.info("В таблице ниже приведена сумма ТРЗ всех ролей по каждому эпику. Эпики с ТРЗ > 40 выделены красным.")

#     # Функция для подсветки превышения порога в 40 ТРЗ
#     def highlight_high_trz(val):
#         return 'background-color: #d9534f; color: white; font-weight: bold' if val > 40 else ''

#     # Выводим таблицу со стилизацией
#     st.dataframe(
#         epic_trz_summary.style.map(highlight_high_trz, subset=['Общий ТРЗ']),
#         use_container_width=True,
#         hide_index=True
#     )
# else:
#     st.warning("В выбранных данных не найдены колонки с ТРЗ для расчета сводки по эпикам.")

# st.write("---")

# ##############################################################################
# # --- Раздел 2.2: Распределение нагрузки по спринтам ---
# st.header("📅 2.2. Нагрузка по спринтам")

# # Отфильтруем пустые спринты
# df_sprints_chart = df_g.dropna(subset=['Номер спринта']).copy()

# if not df_sprints_chart.empty:
#     # 1. ЗАЩИТА ОТ БАГА PLOTLY: Добавляем префикс, чтобы сделать значения 100% текстом
#     df_sprints_chart['Спринт_текст'] = "Спринт " + df_sprints_chart['Номер спринта'].astype(int).astype(str)
    
#     # Группируем данные: Спринт + Роль
#     sprint_stats = df_sprints_chart.groupby(['Спринт_текст', 'Роль'])['ТРЗ'].sum().reset_index()
#     sprint_totals = sprint_stats.groupby('Спринт_текст')['ТРЗ'].sum().reset_index()
    
#     # 2. ПРАВИЛЬНАЯ СОРТИРОВКА: Собираем правильный порядок категорий, исходя из чисел
#     sorted_sprints_nums = sorted(df_sprints_chart['Номер спринта'].astype(int).unique())
#     category_order = [f"Спринт {num}" for num in sorted_sprints_nums]
    
#     # 3. Строим гистограмму
#     fig_sprints = px.bar(
#         sprint_stats,
#         x='Спринт_текст',
#         y='ТРЗ',
#         color='Роль',
#         title="Суммарный объем задач (ТРЗ) в разрезе спринтов и ролей",
#         labels={'ТРЗ': 'Трудозатраты (чел/дн)', 'Спринт_текст': 'Спринт'},
#         template=template_name,
#         text_auto='.1f' 
#     )

#     # Настройка осей: применяем наш жесткий массив сортировки
#     fig_sprints.update_layout(
#         xaxis={
#             'type': 'category', 
#             'categoryorder': 'array', 
#             'categoryarray': category_order # Задаем правильный порядок
#         }, 
#         barmode='stack',
#         legend_title_text='Стек',
#         height=500
#     )

#     # Добавляем общие итоги над столбцами (теперь они 100% приклеятся к центрам столбцов)
#     for _, row in sprint_totals.iterrows():
#         fig_sprints.add_annotation(
#             x=row['Спринт_текст'], 
#             y=row['ТРЗ'],
#             text=f"<b>{row['ТРЗ']:.1f}</b>",
#             showarrow=False,
#             yshift=10,
#             font=dict(color=theme.get("text_color", "white") if theme else "white")
#         )

#     st.plotly_chart(fig_sprints, use_container_width=True)

#     # 4. Краткий вывод/метрика
#     avg_load = sprint_totals['ТРЗ'].mean()
#     st.caption(f"Средняя нагрузка на спринт: **{avg_load:.1f}** чел/дн.")
# else:
#     st.warning("Недостаточно данных для построения гистограммы по спринтам.")

# st.write("---")


# ##############################################################################
# # --- Раздел 3: Аналитика и расчет ресурсов (Capacity Planning) ---
# st.header("📊 3. Аналитика и расчет ресурсов")

# # 1. Задаем период анализа
# st.subheader("🗓️ Период анализа")
# col_p1, col_p2 = st.columns(2)
# with col_p1:
#     analysis_start = st.date_input("Начало периода", value=pd.to_datetime('2026-01-01').date())
# with col_p2:
#     analysis_end = st.date_input("Конец периода", value=pd.to_datetime('2026-03-31').date())

# if analysis_start > analysis_end:
#     st.error("Ошибка: Дата начала не может быть позже даты окончания.")
#     st.stop()

# # Считаем общее кол-во рабочих дней в периоде (5/2)
# total_working_days = np.busday_count(analysis_start, analysis_end + pd.Timedelta(days=1))

# # 2. Подготовка словаря ставок (те, кто не 1.0)
# rates_map = dict(zip(df_rates['Исполнитель'], df_rates['Ставка']))

# # 3. Подготовка праздников
# # Превращаем даты в объекты date для np.busday_count
# df_holidays['Дата_dt'] = pd.to_datetime(df_holidays['Дата'], dayfirst=True, errors='coerce').dt.date
# general_holidays = df_holidays[df_holidays['Исполнитель'].str.lower() == 'все']['Дата_dt'].dropna().tolist()

# # 4. Расчет по стекам
# roles_list = ['Эксперт RnD', 'Аналитик', 'Архитектор', 'UX/UI', 'C#', 'Py', 'React', 'QA','DevOps']
# available_dict = {}
# occupied_dict = {}

# for role in roles_list:
#     role_capacity = 0
    
#     # Проверяем, есть ли такая роль (колонка) на листе Штат
#     if role in df_people.columns:
#         # Получаем список сотрудников для этой роли
#         staff_list = [name for name in df_people[role].unique() if str(name).strip() and str(name).lower() != 'nan']
        
#         for person in staff_list:
#             # А. Определяем СТАВКУ (из листа Ставка или 1.0 по умолчанию)
#             fte = rates_map.get(person, 1.0)
#             try:
#                 fte = float(str(fte).replace(',', '.'))
#             except (ValueError, TypeError):
#                 fte = 1.0
                
#             # Б. Определяем ПРАЗДНИКИ для этого человека (Все + его личные)
#             person_specific_holidays = df_holidays[df_holidays['Исполнитель'] == person]['Дата_dt'].dropna().tolist()
#             combined_holidays = list(set(general_holidays + person_specific_holidays))
            
#             # В. Считаем рабочие дни за вычетом праздников
#             wd_count = np.busday_count(
#                 analysis_start, 
#                 analysis_end + pd.Timedelta(days=1),
#                 holidays=combined_holidays
#             )
            
#             # Г. Вычитаем ОТПУСКА
#             person_vacation_days = 0
#             person_vacations = df_leave[df_leave['Исполнитель'] == person]
#             for _, vac in person_vacations.iterrows():
#                 v_start = pd.to_datetime(vac['НАЧАЛО'], dayfirst=True, errors='coerce').date()
#                 v_end = pd.to_datetime(vac['КОНЕЦ'], dayfirst=True, errors='coerce').date()
#                 if pd.isna(v_start) or pd.isna(v_end): continue
                
#                 overlap_start = max(v_start, analysis_start)
#                 overlap_end = min(v_end, analysis_end)
#                 if overlap_start <= overlap_end:
#                     # ВАЖНО: При расчете дней отпуска тоже учитываем праздники
#                     person_vacation_days += np.busday_count(
#                         overlap_start, 
#                         overlap_end + pd.Timedelta(days=1),
#                         holidays=combined_holidays
#                     )
            
#             # Итоговая мощность человека
#             person_capacity = (wd_count - person_vacation_days) * fte
#             role_capacity += person_capacity
            
#     available_dict[role] = round(role_capacity, 1)

#     # Д. ЗАНЯТОЕ ТРЗ (из бэклога)
#     trz_col = f"трз {role}"
#     if trz_col in df.columns:
#         val = pd.to_numeric(df[trz_col].astype(str).str.replace(',', '.'), errors='coerce').fillna(0).sum()
#         occupied_dict[role] = round(val, 1)
#     else:
#         occupied_dict[role] = 0

# # --- Вывод результатов ---
# summary_records = []
# for role in roles_list:
#     avail = available_dict[role]
#     occ = occupied_dict[role]
#     summary_records.append({
#         "Стек": role,
#         "Доступно (чел/дн)": avail,
#         "Занято (ТРЗ)": occ,
#         "Свободно": round(avail - occ, 1)
#     })

# df_final_stats = pd.DataFrame(summary_records).set_index("Стек").T


# st.subheader("📈 Сводная статистика по ресурсам")
# st.info(f"Период: {analysis_start} — {analysis_end}. Всего рабочих дней в периоде: {total_working_days}")

# # Функция для подсветки дефицита (отрицательных значений)
# def style_negative(v):
#     return 'color: #d9534f; font-weight: bold' if isinstance(v, (int, float)) and v < 0 else None

# st.dataframe(
#     df_final_stats.style.map(style_negative, subset=pd.IndexSlice["Свободно", :]),
#     use_container_width=True
# )

# # Визуальный график дефицита/профицита
# fig_capacity = px.bar(
#     pd.DataFrame(summary_records),
#     x="Стек",
#     y="Свободно",
#     text="Свободно",
#     title="Свободная емкость по стекам (в человеко-днях)",
#     color="Свободно",
#     color_continuous_scale="RdYlGn" # От красного (дефицит) к зеленому (свободно)
# )
# st.plotly_chart(fig_capacity, use_container_width=True)


# # # --- Визуализация: Календарь отсутствий (Отпуска по стекам + Праздники) ---
# # st.subheader("🏖️ Календарь отсутствий и сетка спринтов")

# # # 1. Маппинг Исполнитель -> Роль
# person_to_role = {}
# for role in roles_list:
#     if role in df_people.columns:
#         names = df_people[role].dropna().unique()
#         for n in names:
#             name_str = str(n).strip()
#             if name_str and name_str.lower() != 'nan':
#                 person_to_role[name_str] = role

# # 2. Сбор всех активных сотрудников для отрисовки общих праздников
# all_staff = set()
# for role in roles_list:
#     if role in df_people.columns:
#         all_staff.update([str(n) for n in df_people[role].unique() if str(n).strip() and str(n).lower() != 'nan'])

# # 3. Подготовка данных
# calendar_data = []

# # --- ОТПУСКА (раскрашиваем по стекам) ---
# for _, row in df_leave.iterrows():
#     person = row['Исполнитель']
#     v_start = pd.to_datetime(row['НАЧАЛО'], dayfirst=True, errors='coerce').date()
#     v_end = pd.to_datetime(row['КОНЕЦ'], dayfirst=True, errors='coerce').date()
    
#     if pd.isna(v_start) or pd.isna(v_end): continue
    
#     if v_start <= analysis_end and v_end >= analysis_start:
#         role = person_to_role.get(person, "Вне штата")
#         calendar_data.append(dict(
#             Сотрудник=person,
#             Начало=max(v_start, analysis_start),
#             Конец=min(v_end, analysis_end) + pd.Timedelta(days=1),
#             Тип='Отпуск',
#             Группа_цвета=role, # Для отпусков пишем название стека
#             Детали=f"Отпуск ({role}): {v_start.strftime('%d.%m')} - {v_end.strftime('%d.%m')}"
#         ))

# # --- ПРАЗДНИКИ (общие и личные) ---
# for _, row in df_holidays.iterrows():
#     h_date = row['Дата_dt']
#     target = str(row['Исполнитель']).strip()
#     if pd.isna(h_date): continue
    
#     if analysis_start <= h_date <= analysis_end:
#         if target.lower() == 'все':
#             for person in all_staff:
#                 calendar_data.append(dict(
#                     Сотрудник=person,
#                     Начало=h_date,
#                     Конец=h_date + pd.Timedelta(days=1),
#                     Тип='Праздник',
#                     Группа_цвета='Праздник (Общий)', # Отдельная категория в легенде
#                     Детали=f"Общий праздник: {h_date.strftime('%d.%m')}"
#                 ))
#         else:
#             calendar_data.append(dict(
#                 Сотрудник=target,
#                 Начало=h_date,
#                 Конец=h_date + pd.Timedelta(days=1),
#                 Тип='Праздник',
#                 Группа_цвета='Праздник (Личный)', # Отдельная категория в легенде
#                 Детали=f"Личный праздник: {h_date.strftime('%d.%m')}"
#             ))

# if calendar_data:
#     df_cal = pd.DataFrame(calendar_data)
    
#     # --- ШАГ 1: ЖЕСТКАЯ СОРТИРОВКА ИМЕН ПО СТЕКАМ ---
#     # Собираем всех уникальных людей, которые попали на график
#     unique_people = list(df_cal['Сотрудник'].unique())
    
#     # Сортируем список: сначала по названию стека (из person_to_role), затем по алфавиту
#     unique_people.sort(key=lambda p: (person_to_role.get(p, "ЯЯ_Вне штата"), p))
    
#     # Plotly рисует ось Y снизу вверх, поэтому переворачиваем список, 
#     # чтобы первый стек оказался на самом верху графика
#     sorted_names = list(reversed(unique_people))
    
#     # Настраиваем цвета
#     manual_colors = {
#         'Праздник (Общий)': '#d62728', 
#         'Праздник (Личный)': '#ff7f0e' 
#     }
    
#     fig_cal = px.timeline(
#         df_cal, 
#         x_start="Начало", 
#         x_end="Конец", 
#         y="Сотрудник", 
#         color="Группа_цвета",
#         hover_name="Детали",
#         color_discrete_map=manual_colors,
#         title="Календарь отсутствий: Отпуска по стекам + Праздники",
#         template=template_name
#     )

#     # --- СЕТКА СПРИНТОВ ---
#     if not df_sprint.empty:
#         for _, spr in df_sprint.iterrows():
#             s_start = pd.to_datetime(spr['Дата начала'], dayfirst=True, errors='coerce').date()
#             s_end = pd.to_datetime(spr['Дата окончания'], dayfirst=True, errors='coerce').date()
#             s_num = spr.get('Номер спринта', '?')
            
#             if pd.isna(s_start) or pd.isna(s_end): continue
            
#             if s_start <= analysis_end and s_end >= analysis_start:
#                 fill_color = "rgba(100, 150, 255, 0.08)" if int(s_num or 0) % 2 == 0 else "rgba(100, 150, 255, 0.02)"
#                 fig_cal.add_vrect(x0=s_start, x1=s_end, fillcolor=fill_color, line_width=0, layer="below")
                
#                 fig_cal.add_annotation(
#                     x=s_start + (s_end - s_start)/2, y=1.02, yref="paper",
#                     text=f"Спринт {s_num}", showarrow=False,
#                     font=dict(size=10, color="gray"),
#                     textangle=0 if (s_end - s_start).days > 7 else -90
#                 )

#     # --- ШАГ 2: ПРИМЕНЯЕМ СОРТИРОВКУ К ОСИ Y ---
#     # Убедись, что ниже нет других вызовов update_yaxes!
#     fig_cal.update_yaxes(
#         categoryorder="array", 
#         categoryarray=sorted_names, 
#         title=""
#     )
    
#     fig_cal.update_xaxes(tickformat="%d.%m", range=[analysis_start, analysis_end])
#     fig_cal.update_layout(height=max(450, len(sorted_names) * 30), legend_title_text="Категория")
    
#     st.plotly_chart(fig_cal, use_container_width=True)
# else:
#     st.info("Нет данных для отображения.")


# # --- Аналитика: Пересечение отпусков внутри стека ---
# st.subheader("⚠️ Анализ рисков: Пересечение отпусков")
# st.write("Таблица показывает ситуации, когда два или более сотрудников одного стека уходят в отпуск одновременно.")

# # 1. Собираем все валидные отпуска в периоде с привязанным стеком
# vacations_list = []
# for _, row in df_leave.iterrows():
#     person = row['Исполнитель']
#     v_start = pd.to_datetime(row['НАЧАЛО'], dayfirst=True, errors='coerce').date()
#     v_end = pd.to_datetime(row['КОНЕЦ'], dayfirst=True, errors='coerce').date()

#     if pd.isna(v_start) or pd.isna(v_end): 
#         continue

#     # Если отпуск попадает в период анализа
#     if v_start <= analysis_end and v_end >= analysis_start:
#         role = person_to_role.get(person, "Вне штата")
        
#         # Для расчета дней обрезаем отпуск границами периода анализа
#         eff_start = max(v_start, analysis_start)
#         eff_end = min(v_end, analysis_end)
        
#         vacations_list.append({
#             'Стек': role,
#             'Сотрудник': person,
#             'Начало': eff_start,
#             'Конец': eff_end,
#             'Исходное_начало': v_start,
#             'Исходный_конец': v_end
#         })

# df_vac = pd.DataFrame(vacations_list)

# overlaps = []
# if not df_vac.empty:
#     # 2. Группируем по стекам и ищем пересечения
#     for role, group in df_vac.groupby('Стек'):
#         if len(group) < 2:
#             continue # Пересечений быть не может, если отпуск только один
            
#         # Превращаем в список словарей для удобного попарного сравнения
#         records = group.to_dict('records')
        
#         # Сравниваем каждый отпуск с каждым
#         for i in range(len(records)):
#             for j in range(i + 1, len(records)):
#                 v1 = records[i]
#                 v2 = records[j]

#                 # Нас не интересуют два разных отпуска одного и того же человека
#                 if v1['Сотрудник'] == v2['Сотрудник']:
#                     continue

#                 # Формула проверки пересечения двух отрезков: Начало1 <= Конец2 И Начало2 <= Конец1
#                 if v1['Начало'] <= v2['Конец'] and v2['Начало'] <= v1['Конец']:
#                     # Находим границы самого пересечения
#                     o_start = max(v1['Начало'], v2['Начало'])
#                     o_end = min(v1['Конец'], v2['Конец'])
                    
#                     # Считаем, сколько это РАБОЧИХ дней (с учетом общих праздников)
#                     overlap_days = np.busday_count(
#                         o_start, 
#                         o_end + pd.Timedelta(days=1), 
#                         holidays=general_holidays
#                     )

#                     # Если пересечение выпадает только на выходные/праздники, пропускаем
#                     if overlap_days > 0:
#                         overlaps.append({
#                             'Стек': role,
#                             'Сотрудник 1': v1['Сотрудник'],
#                             'Период 1': f"{v1['Исходное_начало'].strftime('%d.%m')} - {v1['Исходный_конец'].strftime('%d.%m')}",
#                             'Сотрудник 2': v2['Сотрудник'],
#                             'Период 2': f"{v2['Исходное_начало'].strftime('%d.%m')} - {v2['Исходный_конец'].strftime('%d.%m')}",
#                             'Дни пересечения': f"{o_start.strftime('%d.%m')} - {o_end.strftime('%d.%m')}",
#                             'Раб. дней риска': overlap_days
#                         })

# # 3. Выводим результат
# if overlaps:
#     df_overlaps = pd.DataFrame(overlaps)
    
#     # Сортируем: сначала по стеку, затем по убыванию дней риска
#     df_overlaps = df_overlaps.sort_values(by=['Стек', 'Раб. дней риска'], ascending=[True, False])
    
#     st.error(f"Внимание! Обнаружено {len(df_overlaps)} пересечений отпусков у сотрудников одного стека.")
    
#     # Функция для подсветки опасных пересечений (больше 3 дней)
#     def highlight_risk(val):
#         color = '#d9534f' if isinstance(val, (int, float)) and val > 3 else 'white'
#         return f'color: {color}; font-weight: bold'
        
#     st.dataframe(
#         df_overlaps.style.map(highlight_risk, subset=['Раб. дней риска']), 
#         use_container_width=True, 
#         hide_index=True
#     )
# else:
#     st.success("Отлично! Критичных пересечений отпусков у сотрудников одного стека в выбранном периоде не обнаружено.")


# # --- Аналитика: Индивидуальная загрузка каждого сотрудника ---
# st.subheader("👤 Индивидуальная загрузка сотрудников")
# st.write("Детальная разбивка по каждому члену команды: доступная емкость, нагрузка задачами и количество дней отпуска.")

# # 1. Считаем суммарное ЗАНЯТОЕ ТРЗ для каждого человека
# # Поскольку в df_g поле 'ТРЗ' уже было приведено к числовому формату ранее в коде, мы можем просто сгруппировать его:
# person_occupied = df_g.groupby('Исполнитель')['ТРЗ'].sum().to_dict()

# individual_stats = []

# # 2. Проходим по всем сотрудникам штата и считаем их личные показатели
# for role in roles_list:
#     if role in df_people.columns:
#         staff_list = [name for name in df_people[role].unique() if str(name).strip() and str(name).lower() != 'nan']
#         for person in staff_list:
#             # А. Получаем ставку
#             fte = rates_map.get(person, 1.0)
#             try: 
#                 fte = float(str(fte).replace(',', '.'))
#             except (ValueError, TypeError): 
#                 fte = 1.0
            
#             # Б. Собираем праздники
#             person_specific_holidays = df_holidays[df_holidays['Исполнитель'] == person]['Дата_dt'].dropna().tolist()
#             combined_holidays = list(set(general_holidays + person_specific_holidays))
            
#             # В. Считаем общие рабочие дни в периоде для этого человека
#             wd_count = np.busday_count(
#                 analysis_start, 
#                 analysis_end + pd.Timedelta(days=1),
#                 holidays=combined_holidays
#             )
            
#             # Г. Считаем рабочие дни ОТПУСКА в выбранном периоде
#             person_vacation_days = 0
#             person_vacations = df_leave[df_leave['Исполнитель'] == person]
#             for _, vac in person_vacations.iterrows():
#                 v_start = pd.to_datetime(vac['НАЧАЛО'], dayfirst=True, errors='coerce').date()
#                 v_end = pd.to_datetime(vac['КОНЕЦ'], dayfirst=True, errors='coerce').date()
#                 if pd.isna(v_start) or pd.isna(v_end): continue
                
#                 overlap_start = max(v_start, analysis_start)
#                 overlap_end = min(v_end, analysis_end)
#                 if overlap_start <= overlap_end:
#                     person_vacation_days += np.busday_count(
#                         overlap_start, 
#                         overlap_end + pd.Timedelta(days=1),
#                         holidays=combined_holidays
#                     )
            
#             # Д. Итоговая математика
#             capacity = (wd_count - person_vacation_days) * fte
#             occupied = person_occupied.get(person, 0.0)
#             free = capacity - occupied
            
#             individual_stats.append({
#                 'Сотрудник': person,
#                 'Стек': role,
#                 'Ставка': fte,
#                 'Дней отпуска': person_vacation_days,
#                 'Доступно ТРЗ': round(capacity, 1),
#                 'Занято ТРЗ': round(occupied, 1),
#                 'Свободно ТРЗ': round(free, 1)
#             })

# # 3. Вывод таблицы и визуализации
# if individual_stats:
#     df_indiv = pd.DataFrame(individual_stats)
    
#     # Сортируем: сначала по стеку, потом по алфавиту имени
#     df_indiv = df_indiv.sort_values(by=['Стек', 'Сотрудник'])
    
#     # Выводим таблицу с нашей функцией style_negative (подсветка перегруза)
#     st.dataframe(
#         df_indiv.style.map(style_negative, subset=['Свободно ТРЗ']),
#         use_container_width=True,
#         hide_index=True
#     )
    
#     # 4. Визуализация: Барчарт индивидуальной нагрузки
#     fig_indiv = px.bar(
#         df_indiv,
#         x="Сотрудник",
#         y=["Занято ТРЗ", "Свободно ТРЗ"],
#         title="Индивидуальная загрузка (в человеко-днях)",
#         color_discrete_map={"Занято ТРЗ": "#ef5350", "Свободно ТРЗ": "#66bb6a"},
#         hover_data={"Стек": True, "Дней отпуска": True}
#     )
    
#     # Настраиваем отображение: relative позволяет свободному ТРЗ уходить в минус на графике
#     fig_indiv.update_layout(
#         barmode='relative', 
#         yaxis_title="ТРЗ (дней)", 
#         xaxis_title="",
#         legend_title_text="Категория"
#     )
    
#     st.plotly_chart(fig_indiv, use_container_width=True)


# # --- Аналитика: Прямые конфликты (Задачи vs Отпуска) ---
# st.subheader("🚩 Конфликты: Задачи в период отпуска")
# st.write("В этой таблице собраны конкретные задачи, даты выполнения которых пересекаются с отпуском исполнителя.")

# task_conflicts = []

# # Проходим по всем рассчитанным задачам в Ганте
# for _, task in df_g.iterrows():
#     person = task['Исполнитель']
#     # Пропускаем, если исполнитель не назначен
#     if not person or str(person).lower() == 'nan':
#         continue
        
#     t_start = task['Дата начала'].date() if isinstance(task['Дата начала'], pd.Timestamp) else task['Дата начала']
#     t_end = task['Дата окончания'].date() if isinstance(task['Дата окончания'], pd.Timestamp) else task['Дата окончания']
    
#     # Ищем отпуска этого конкретного человека
#     person_vacations = df_leave[df_leave['Исполнитель'] == person]
    
#     for _, vac in person_vacations.iterrows():
#         v_start = pd.to_datetime(vac['НАЧАЛО'], dayfirst=True, errors='coerce').date()
#         v_end = pd.to_datetime(vac['КОНЕЦ'], dayfirst=True, errors='coerce').date()
        
#         if pd.isna(v_start) or pd.isna(v_end):
#             continue
            
#         # Проверяем пересечение интервалов задачи и отпуска
#         if t_start <= v_end and v_start <= t_end:
#             # Считаем количество рабочих дней конфликта
#             c_start = max(t_start, v_start)
#             c_end = min(t_end, v_end)
            
#             # Используем уже имеющиеся в коде праздники для точности
#             person_specific_holidays = df_holidays[df_holidays['Исполнитель'] == person]['Дата_dt'].dropna().tolist()
#             combined_h = list(set(general_holidays + person_specific_holidays))
            
#             conflict_days = np.busday_count(
#                 c_start, 
#                 c_end + pd.Timedelta(days=1), 
#                 holidays=combined_h
#             )
            
#             if conflict_days > 0:
#                 task_conflicts.append({
#                     'Стек': task['Роль'],
#                     'Исполнитель': person,
#                     'Эпик': task['Название крупная задача (ЭПИК)'], # <-- Добавлено название эпика
#                     'Спринт': task['Номер спринта'],
#                     'Задача': task['Название задачи'],
#                     'Сроки задачи': f"{t_start.strftime('%d.%m')} - {t_end.strftime('%d.%m')}",
#                     'Сроки отпуска': f"{v_start.strftime('%d.%m')} - {v_end.strftime('%d.%m')}",
#                     'Дней конфликта': conflict_days
#                 })

# if task_conflicts:
#     df_task_conf = pd.DataFrame(task_conflicts)
#     # Сортируем по степени риска (где больше всего потерянных дней)
#     df_task_conf = df_task_conf.sort_values(by='Дней конфликта', ascending=False)
    
#     st.warning(f"Найдено {len(df_task_conf)} задач, требующих переноса сроков или смены исполнителя.")
    
#     # Стилизация: подсвечиваем критичные задачи (где конфликт более 2 дней)
#     def highlight_task_risk(v):
#         return 'color: #d9534f; font-weight: bold' if v > 2 else ''

#     st.dataframe(
#         df_task_conf.style.map(highlight_task_risk, subset=['Дней конфликта']),
#         use_container_width=True,
#         hide_index=True
#     )
# else:
#     st.success("Все задачи в текущем плане обеспечены присутствием исполнителей.")




# # -------- Свод по задачам (Эпик/column_task_name): сумма ТРЗ по стекам и Итого --------
# try:
#     _roles = ['Эксперт RnD','Аналитик','Архитектор','UX/UI','C#','Py','React','QA']
#     _trz_cols = [f"трз {r}" for r in _roles if f"трз {r}" in df.columns]
#     if _trz_cols:
#         _num = df[_trz_cols].replace(',', '.', regex=True).apply(pd.to_numeric, errors='coerce').fillna(0)
#         _tmp = df[[column_epic_name]].copy()
#         _tmp[_trz_cols] = _num
#         task_sum = _tmp.groupby(column_epic_name, dropna=False)[_trz_cols].sum().reset_index()
#         task_sum["Итого"] = task_sum[_trz_cols].sum(axis=1)
#         task_sum = task_sum.sort_values("Итого", ascending=False)
#         task_sum.insert(1, "Итого", task_sum.pop("Итого"))

#         st.subheader("Сумма ТРЗ по эпикам")
#         highlight = JsCode("""
#             function(params) {
#                 let v = params.value;
#                 if (v > 40) {return {'color':'white','backgroundColor':'#d9534f'};}
#                 if (v >= 35 && v <= 49) {return {'backgroundColor':'#f0ad4e'};}
#                 return {};
#             }
#         """)
#         gb_tasks = GridOptionsBuilder.from_dataframe(task_sum)
#         gb_tasks.configure_default_column(resizable=True, sortable=True)
#         gb_tasks.configure_column('Итого')#, cellStyle=highlight)
#         AgGrid(task_sum, gridOptions=gb_tasks.build(), height=320, update_mode=GridUpdateMode.NO_UPDATE)
# except Exception as e:
#     st.warning(f"Не удалось построить свод по задачам: {e}")

# df_f = df_g.dropna(subset=['Дата начала', 'Дата окончания']).copy()

# col_start, col_end = st.columns(2)
# with col_start:
#     period_start = st.date_input(
#         "Период: начало",
#         value=pd.to_datetime('01.01.2026').date()#pd.to_datetime(df_f['Дата начала'].min()).date()
#     )
# with col_end:
#     period_end   = st.date_input(
#         "Период: конец",
#         value=pd.to_datetime('31.12.2026').date()#pd.to_datetime(df_f['Дата окончания'].max()).date()
#     )
# # df_f['Start_calc'] = pd.to_datetime(df_f['Дата начала'], errors='coerce')
# # df_f['Finish'] = pd.to_datetime(df_f['Дата окончания'], errors='coerce')
# df_f = df_f[(df_f['Дата начала'].dt.date >= period_start) &
#             (df_f['Дата окончания'].dt.date <= period_end)]
        

# # Вычисление свободной ёмкости по периодам с учётом отпусков
# st.subheader("Свободная ёмкость исполнителей в период")
# records_cap = []
# for p in df_f['Исполнитель'].unique():
#     # всего рабочих дней в выбранном периоде
#     total_bd = np.busday_count(period_start, period_end + pd.Timedelta(days=1))
#     # считаем дни отпуска пересечением с периодом
#     vac_days = 0
#     for _, v in df_leave[df_leave['Исполнитель'] == p].iterrows():
#         vs_ts = pd.to_datetime(v['НАЧАЛО'], errors="coerce")
#         ve_ts = pd.to_datetime(v['КОНЕЦ'], errors="coerce")

#     # если дата начала или конца отсутствует — пропускаем запись
#         if pd.isna(vs_ts) or pd.isna(ve_ts):
#             continue

#         vs = vs_ts.date()
#         ve = ve_ts.date()

#         start_int = max(vs, period_start)
#         end_int   = min(ve, period_end)
#         if start_int <= end_int:
#             vac_days += np.busday_count(start_int, end_int + pd.Timedelta(days=1))
#     # суммарная нагрузка по задачам (ТРЗ в днях)
#     task_bd = df_f[df_f['Исполнитель'] == p]['ТРЗ'].sum()
#     free_bd = total_bd - vac_days - task_bd

#     records_cap.append({
#         'Исполнитель': p,
#         'Роль': df_f[df_f['Исполнитель']==p]['Роль'].mode()[0],
#         'Всего раб. дн.': total_bd,
#         'Дн. отпуска': vac_days,
#         'Занято (ТРЗ дн.)': task_bd,
#         'Доступно (дн.)': free_bd
#     })

# df_res = pd.DataFrame(records_cap)


# # Выводим в AgGrid
# gb_cap = GridOptionsBuilder.from_dataframe(df_res)
# gb_cap.configure_default_column(resizable=True, sortable=True)
# AgGrid(
#     df_res,
#     gridOptions=gb_cap.build(),
#     fit_columns_on_grid_load=True,
#     height=300,
#     key="capacity_aggrid"
# )

# # st.write(df_res.sort_values('Доступно (дн.)'))

# #  Горизонтальная диаграмма «Нагрузка vs Отпуск vs Свободно»
# fig = px.bar(
#     df_res.sort_values('Доступно (дн.)'),
#     y='Исполнитель',
#     x=['Занято (ТРЗ дн.)','Дн. отпуска','Доступно (дн.)'],
#     orientation='h',
#     title="Нагрузка vs Отпуск vs Свободное время",
#     labels={'value':'Дней','variable':'Категория'}
# )
# bar_px=30
# fig.update_layout(barmode='stack', height=len(df_res) * bar_px)
# st.plotly_chart(fig, use_container_width=True)

# # Стековая диаграмма объёма задач по ролям
# pivot = (
#     df_f
#     .pivot_table(
#         index='Роль',
#         columns='Исполнитель',
#         values='ТРЗ',
#         aggfunc='sum',
#         fill_value=0
#     )
#     .reset_index()
# )
# fig = px.bar(
#     pivot,
#     x='Роль',
#     y=pivot.columns.drop('Роль'),
#     title="Объём задач по ролям и исполнителям"
# )
# fig.update_layout(barmode='stack', height=400, xaxis_title=None)
# st.plotly_chart(fig, use_container_width=True)


# # Тепловая карта ежедневной загрузки и отпуска

# # st.write(df_f)
# # Сформируем матрицу: index=Исполнитель, cols=даты спринта
# df_f = df_f#.dropna()
# dates = pd.date_range(
#     df_f['Дата начала'].dropna().min(),
#     df_f['Дата окончания'].dropna().max(),
#     freq='D'
# )
# persons = df_f['Исполнитель'].unique()
# cal = pd.DataFrame(0, index=persons, columns=dates)

# # Нагрузка
# for _, r in df_f.iterrows():
#     p = r['Исполнитель']
#     start, end = pd.to_datetime(r['Дата начала'],format='%d.%m.%Y',dayfirst=True,), pd.to_datetime(r['Дата окончания'],format='%d.%m.%Y',dayfirst=True,)
#     for d in pd.date_range(start, end, freq='D'):
#         if d.weekday()<5:
#             cal.at[p, d] += 1

# # Отпуска пометим -1
# for _, r in df_leave.iterrows():
#     p = r['Исполнитель']
#     start = pd.to_datetime(r['НАЧАЛО'],format='%d.%m.%Y',dayfirst=True, errors="coerce")
#     end   = pd.to_datetime(r['КОНЕЦ'],format='%d.%m.%Y',dayfirst=True, errors="coerce")

#     # если нет начала или конца — пропускаем запись
#     if pd.isna(start) or pd.isna(end):
#         continue

#     for d in pd.date_range(start, end, freq='D'):
#         if d.weekday() < 5 and p in cal.index and d in cal.columns:
#             cal.at[p, d] = -1
#     # for d in pd.date_range(r['НАЧАЛО'], r['КОНЕЦ'], freq='D'):
#     #     if d.weekday()<5 and p in cal.index and d in cal.columns:
#     #         cal.at[p, d] = -1

# fig = px.imshow(
#     cal,
#     labels=dict(x="Дата", y="Исполнитель", color="Нагрузка"),
#     x=cal.columns,
#     y=cal.index,
#     title="Ежедневная загрузка (и отпуска)"
# )
# # Перекрашиваем отпуск в серый:
# fig.update_traces(
#     zmin=-1, zmax=cal.values.max(),
#     colorscale=[
#         [0.0, "lightgray"], [0.01, "white"],
#         [0.5, "lightblue"], [1.0, "blue"]
#     ]
# )
# st.plotly_chart(fig, use_container_width=True, height=500)

# # Объём задач по спринтам
# sprint_vol = (
#     df_f
#     .groupby('Номер спринта')['ТРЗ']
#     .sum()
#     .reset_index(name='Объём TRЗ')
# )

# fig = px.bar(
#     sprint_vol,
#     x='Номер спринта',
#     y='Объём TRЗ',
#     title="Объём задач по спринтам"
# )
# st.plotly_chart(fig, use_container_width=True, height=350)

# # 6. Таблица конфликтов задач с отпусками
# conflicts = []
# for _, r in df_f.iterrows():
#     p = r['Исполнитель']

#     s = pd.to_datetime(r['Дата начала'], errors="coerce")
#     e = pd.to_datetime(r['Дата окончания'], errors="coerce")

#     # если интервал задачи битый — пропускаем
#     if pd.isna(s) or pd.isna(e):
#         continue

#     leaves = df_leave[df_leave['Исполнитель'] == p].copy()

#     leaves['НАЧАЛО'] = pd.to_datetime(leaves['НАЧАЛО'],format='%d.%m.%Y',dayfirst=True,  errors="coerce")
#     leaves['КОНЕЦ']  = pd.to_datetime(leaves['КОНЕЦ'],format='%d.%m.%Y',dayfirst=True,  errors="coerce")

#     if any(
#         not (e < ld or s > lu)
#         for ld, lu in zip(leaves['НАЧАЛО'], leaves['КОНЕЦ'])
#         if pd.notna(ld) and pd.notna(lu)
#     ):
#         conflicts.append(r)
# # for _, r in df_f.iterrows():
# #     p = r['Исполнитель']
# #     s, e = pd.to_datetime(r['Дата начала']), pd.to_datetime(r['Дата окончания'])
# #     leaves = df_leave[df_leave['Исполнитель']==p]
# #     if any(not (e < ld or s > lu)
# #            for ld, lu in zip(leaves['НАЧАЛО'], leaves['КОНЕЦ'])):
# #         conflicts.append(r)

# df_conf = pd.DataFrame(conflicts)
# st.subheader("Задачи, пересекающиеся с отпусками")
# if not df_conf.empty:
#     gb2 = GridOptionsBuilder.from_dataframe(df_conf)
#     gb2.configure_default_column(resizable=True, sortable=True)
#     AgGrid(df_conf, gridOptions=gb2.build(), height=300)
# else:
#     st.success("Конфликтов не обнаружено")


# ##############################################################################
# # Калькулятор рабочих часов ---
# with st.expander("Калькулятор рабочих часов", False):
#     column1, column2,column3 = st.columns(3)
#     with column1:
#         start_date = st.date_input("Дата начала", value=pd.to_datetime("today").date())
#     with column3:
#         end_date = st.date_input("Дата окончания", value=pd.to_datetime("today").date())
#         if start_date > end_date:
#             st.error("❗ Дата начала должна быть раньше или равна дате окончания.")
#         else:
#             with column1:
#                 hours_per_day = st.number_input("Рабочих часов в дне", min_value=1, max_value=24, value=8)
#             total_hours = working_hours_between(start_date, end_date, hours_per_day)
#             with column2:
#                 # st.write("### Статистика по кварталам")
#                 # st.write("В Q3 ", working_hours_between('2025-06-30', '2025-08-31', 8), 'рабочих часов')
#                 # st.write("В Q4 ", working_hours_between('2025-09-01', '2025-12-31', 8), 'рабочих часов')
#                 st.success(f"Между {start_date} и {end_date} включая оба дня:\n"
#                             f"• рабочих дней: {total_hours // hours_per_day}\n"
#                             f"• рабочих часов: {total_hours}")