import streamlit.components.v1 as components_html
import streamlit as st
import pandas as pd
from pandas.tseries.offsets import BusinessDay
import plotly.express as px
from utils import wrap_text
import plotly.graph_objects as go
from utils import assign_tracks

import streamlit as st

def ui_gantt_settings(
    df,
    prefix: str,
    title: str = "⚙️ Настройки графика",
    # индексы по умолчанию для selectbox (под твой пример)
    default_task_idx: int = 3,
    default_y_idx: int = 2,
    default_color_idx: int = 0,
    # фиксированные имена для дат как в твоём коде
    default_start_col: str = "Дата начала",
    default_end_col: str = "Дата окончания",
    # дефолты слайдеров
    default_max_len: int = 80,
    default_font_size: int = 8,
    default_width: int = 1700,
    default_height: int = 1200,
    # дефолты прочего
    default_swap_text: bool = True,
    default_sort_first: str = "Start",
    default_asc_desc: str = "ASC",
):
    """Возвращает dict с выбранными настройками. Можно вызывать многократно с разными prefix."""
    cols_all = df.columns.tolist()
    # безопасно ограничим индексы
    def _safe_idx(i): return max(0, min(i, len(cols_all)-1)) if cols_all else 0

    with st.expander(title, expanded=False):
        with st.form(f"{prefix}_gantt_settings"):
            col1, col2, col3,col4,col5 = st.columns(5)

            with col1:
                st.write("Фильтры")
                task_col  = st.selectbox("Подпись в тултипе", cols_all, index=_safe_idx(default_task_idx), key=f"{prefix}_task")
                # y_axis    = st.selectbox("Группировать по (ось Y)", cols_all, index=_safe_idx(default_y_idx), key=f"{prefix}_y")
                color_by  = st.selectbox("Окрашивать по", cols_all, index=_safe_idx(default_color_idx), key=f"{prefix}_color")

                # фиксированные поля дат, как у тебя
                start_col = default_start_col
                end_col   = default_end_col

            with col2:
                font_size  = st.slider("Размер текста", min_value=1, max_value=50, value=default_font_size, key=f"{prefix}_fsz")
                width_plot = st.slider("Ширина диаграммы", min_value=1500, max_value=3000, value=default_width, key=f"{prefix}_w")
                height_plot= st.slider("Высота диаграммы", min_value=500, max_value=2500, value=default_height, key=f"{prefix}_h")

            with col3:
                sort_candidates = ['Task','Start','Finish','Y_Group','Resource']
                # подстрахуемся: если таких колонок нет в df, всё равно дадим выбрать
                sort_val_first = st.selectbox("Сортировать задачи по", sort_candidates, index=sort_candidates.index(default_sort_first), key=f"{prefix}_sort1")
                asc_desc = st.selectbox("Направление", ['ASC','DESC'], index=0 if default_asc_desc=='ASC' else 1, key=f"{prefix}_asc")
            with col4:
                swap_text = st.checkbox("Включить перенос слов", value=default_swap_text, key=f"{prefix}_wrap")
                max_len    = st.slider("Длина подписей", min_value=10, max_value=100, value=default_max_len, key=f"{prefix}_maxlen")
            with col5:
                fit_names = st.checkbox("Вписать подписи в бары", value=True, key=f"{prefix}_fit")
                font_size_names = st.slider("Размер шрифта подписей", min_value=1, max_value=30, value=8, key=f"{prefix}_fsz_names")
                

            submitted = st.form_submit_button("Применить", use_container_width=True, type="primary")

    return {
        "task_col": task_col,
        # "y_axis": y_axis,
        "color_by": color_by,
        "start_col": start_col,
        "end_col": end_col,
        "max_len": max_len,
        "font_size": font_size,
        "width_plot": width_plot,
        "height_plot": height_plot,
        "swap_text": swap_text,
        "sort_val_first": sort_val_first,
        "asc_desc": asc_desc,
        "fit_names": fit_names,
        "font_size_names": font_size_names,
        "submitted": submitted,
        # вспомогательные ключи (удобно дальше):
        "prefix": prefix,
        "columns_all": cols_all,
    }


# 🎨 Каталог тем
def get_theme(name: str) -> dict:
    name = name.lower()

    THEMES = {
        # 🌙 Темные
        "soft_dark": {
            "palette": ["#8AB4F8","#F28B82","#81C995","#FDD663","#CF9FFF",
                        "#78D9EC","#E8EAED","#A7FF83","#FFB3C1","#B0BEC5"],
            "bg_paper": "#0F1115",
            "bg_plot":  "#0F1115",
            "grid_color": "rgba(232,234,237,0.10)",
            "text_color": "#E8EAED",
            "zebra_colors": ("rgba(255,255,255,0.05)", "rgba(255,255,255,0.00)"),
            "sprint_fill": "rgba(120,200,255,0.18)",
            "bar_opacity": 0.95,
            "bar_text_color": "#FFFFFF",
            "legend_bg": "rgba(255,255,255,0.03)"
        },
        "midnight": {
            "palette": ["#4E79A7","#F28E2B","#E15759","#76B7B2","#59A14F",
                        "#EDC948","#B07AA1","#FF9DA7","#9C755F","#BAB0AC"],
            "bg_paper": "#0D1B2A",
            "bg_plot":  "#0D1B2A",
            "grid_color": "rgba(255,255,255,0.12)",
            "text_color": "#E0E1DD",
            "zebra_colors": ("rgba(255,255,255,0.06)", "rgba(255,255,255,0.00)"),
            "sprint_fill": "rgba(80,150,255,0.18)",
            "bar_opacity": 0.9,
            "bar_text_color": "#FFFFFF",
            "legend_bg": "rgba(255,255,255,0.04)"
        },
        "mocha": {
            "palette": ["#B08968","#E6B8A2","#A44A3F","#7E6B5A","#C97C5D",
                        "#936639","#FFD9C0","#8C5E58","#D4A373","#EAD2AC"],
            "bg_paper": "#2B2119",
            "bg_plot":  "#2B2119",
            "grid_color": "rgba(255,255,255,0.08)",
            "text_color": "#F8F1E9",
            "zebra_colors": ("rgba(255,255,255,0.04)", "rgba(255,255,255,0.00)"),
            "sprint_fill": "rgba(233,196,106,0.20)",
            "bar_opacity": 0.95,
            "bar_text_color": "#FFF3E1",
            "legend_bg": "rgba(255,255,255,0.05)"
        },
        "ice": {
            "palette": ["#A5C9CA","#395B64","#E7F6F2","#2C3333","#6A9AB0",
                        "#5BC0BE","#3A506B","#1C2541","#5EAAA8","#A3D2CA"],
            "bg_paper": "#1B262C",
            "bg_plot":  "#1B262C",
            "grid_color": "rgba(255,255,255,0.15)",
            "text_color": "#EAF6F6",
            "zebra_colors": ("rgba(255,255,255,0.05)", "rgba(255,255,255,0.00)"),
            "sprint_fill": "rgba(91,192,190,0.20)",
            "bar_opacity": 0.92,
            "bar_text_color": "#FFFFFF",
            "legend_bg": "rgba(255,255,255,0.05)"
        },
        "sunset": {
            "palette": ["#FF6F61","#FFB347","#FFD166","#EF476F","#FF9F1C",
                        "#FDCB82","#F08080","#E5989B","#FFBA93","#F8AFA6"],
            "bg_paper": "#2E1A1A",
            "bg_plot":  "#2E1A1A",
            "grid_color": "rgba(255,255,255,0.1)",
            "text_color": "#FFEFEF",
            "zebra_colors": ("rgba(255,255,255,0.04)", "rgba(255,255,255,0.00)"),
            "sprint_fill": "rgba(255,160,122,0.25)",
            "bar_opacity": 0.9,
            "bar_text_color": "#FFFFFF",
            "legend_bg": "rgba(255,255,255,0.04)"
        },
        "forest": {
            "palette": ["#2E8B57","#A9BA9D","#4E6C50","#739072","#A4BE7B",
                        "#3C6255","#61876E","#9EAD86","#52734D","#B0C592"],
            "bg_paper": "#16241C",
            "bg_plot":  "#16241C",
            "grid_color": "rgba(255,255,255,0.08)",
            "text_color": "#E8F5E9",
            "zebra_colors": ("rgba(255,255,255,0.04)", "rgba(255,255,255,0.00)"),
            "sprint_fill": "rgba(100,180,100,0.22)",
            "bar_opacity": 0.95,
            "bar_text_color": "#FFFFFF",
            "legend_bg": "rgba(255,255,255,0.05)"
        },

        # ☀️ Светлые
        "pastel": {
            "palette": ["#AEC6CF","#FFB347","#77DD77","#FF6961","#F49AC2",
                        "#CFCFC4","#FDFD96","#84B6F4","#FDDB98","#DEA5A4"],
            "bg_paper": "#FFFFFF",
            "bg_plot":  "#FFFFFF",
            "grid_color": "rgba(0,0,0,0.08)",
            "text_color": "#2E2E2E",
            "zebra_colors": ("rgba(0,0,0,0.04)", "rgba(0,0,0,0.00)"),
            "sprint_fill": "rgba(255,182,193,0.25)",
            "bar_opacity": 0.9,
            "bar_text_color": "#2E2E2E",
            "legend_bg": "rgba(0,0,0,0.03)"
        },
        "corporate": {
            "palette": ["#005EB8","#7D3C98","#FF6F00","#00897B","#AFB42B",
                        "#6D4C41","#0288D1","#C2185B","#512DA8","#F9A825"],
            "bg_paper": "#F9F9F9",
            "bg_plot":  "#FFFFFF",
            "grid_color": "rgba(0,0,0,0.1)",
            "text_color": "#212121",
            "zebra_colors": ("rgba(0,0,0,0.03)", "rgba(0,0,0,0.00)"),
            "sprint_fill": "rgba(0,94,184,0.15)",
            "bar_opacity": 0.9,
            "bar_text_color": "#000000",
            "legend_bg": "rgba(0,0,0,0.04)"
        },
        "minimal_white": {
            "palette": ["#264653","#2A9D8F","#E9C46A","#F4A261","#E76F51",
                        "#8AB17D","#D3D3D3","#457B9D","#F28482","#84A59D"],
            "bg_paper": "#FFFFFF",
            "bg_plot":  "#FFFFFF",
            "grid_color": "rgba(0,0,0,0.08)",
            "text_color": "#111111",
            "zebra_colors": ("rgba(0,0,0,0.03)", "rgba(0,0,0,0.00)"),
            "sprint_fill": "rgba(0,0,0,0.06)",
            "bar_opacity": 0.92,
            "bar_text_color": "#111111",
            "legend_bg": "rgba(0,0,0,0.02)"
        },
    }

    if name in THEMES:
        return THEMES[name]

    raise ValueError(f"Неизвестная тема '{name}'. Доступные: {list(THEMES.keys())}")
DEFAULT_THEME = get_theme('SOFT_DARK')  # делаем её дефолтом

def available_themes():
    """Список ключей тем, которые понимает get_theme()."""
    return [
        # тёмные
        "soft_dark", "midnight", "mocha", "ice", "sunset", "forest",
        # светлые
        "pastel", "corporate", "minimal_white"
    ]

DARK_THEMES = {"soft_dark", "midnight", "mocha", "ice", "sunset", "forest"}

def theme_to_template(theme_key: str) -> str:
    """Подбираем plotly template под тему."""
    return "plotly_dark" if theme_key in DARK_THEMES else "simple_white"

def _palette_preview_html(palette):
    # рисуем мини-превью палитры, чтобы было видно цвета в выпадашке
    boxes = "".join(
        f'<span style="display:inline-block;width:16px;height:16px;border-radius:3px;'
        f'margin-right:4px;background:{c};border:1px solid rgba(0,0,0,0.15)"></span>'
        for c in palette[:10]
    )
    return f'<div style="display:flex;gap:4px;align-items:center">{boxes}</div>'

def ui_theme_picker(expanded=False, default_key="soft_dark",suffix=''):
    """Рисует expander с selectbox выбора темы. Возвращает (theme_key, theme_dict, template_name)."""
    with st.expander("🎨 Тема оформления", expanded=expanded):
        # Читаем текущий выбор из session_state (чтобы сохранялся между перерисовками)
        default_idx = max(0, available_themes().index(default_key)) if default_key in available_themes() else 0
        theme_key = st.selectbox(
            "Выберите тему",
            options=available_themes(),
            index=default_idx,
            key="theme_picker_key"+suffix,
        )

        # Получаем словарь темы и template
        theme = get_theme(theme_key)
        template_name = theme_to_template(theme_key)

        # Мини-превью палитры
        st.markdown(
            _palette_preview_html(theme.get("palette", [])),
            unsafe_allow_html=True
        )

        # Подсказка по фону/тексту
        st.caption(f"Фон: {theme.get('bg_plot')}  •  Текст: {theme.get('text_color')}  •  Template: {template_name}")

    return theme_key, theme, template_name


def build_color_map(values, theme, override=None):
    pal = theme.get("palette", DEFAULT_THEME["palette"])
    base = {v: pal[i % len(pal)] for i, v in enumerate(values)}
    if override:
        base.update({k: override[k] for k in override})
    return base

def add_date_markers(
    fig: go.Figure,
    markers: list[dict],
    *,
    line_color="#FF4D4D",
    line_width=2,
    line_dash="dash",        # "solid" | "dash" | "dot" | "dashdot"
    label_color="#FF4D4D",
    label_font_size=14,
    label_y="above",         # "above" (над графиком) | "inside" (вверху внутри)
    label_textangle=90,     # -90 → вертикальная подпись
):
    """
    markers: список словарей {"date": <datetime|str>, "text": "Релиз 1.2"}
    Рисует вертикальные линии на указанных датах на всей высоте графика.
    """
    for m in markers:
        d = pd.to_datetime(m["date"])
        t = str(m.get("text", d.strftime("%d.%m.%Y")))

        # Линия на всю высоту области графика
        fig.add_shape(
            type="line",
            x0=d, x1=d,
            xref="x",
            y0=0, y1=1,
            yref="paper",
            line=dict(color=line_color, width=line_width, dash=line_dash),
            layer="above"  # чтобы было поверх зебры/баров (можешь поменять на "below")
        )

        # Подпись
        if label_y == "inside":
            y, yanchor = 1.0, "auto"    # внутри области графика, у верхней кромки
        else:
            y, yanchor = 1.01, "bottom"  # над графиком — нужен чуть больший margin.t

        fig.add_annotation(
            x=d, y=y,
            xref="x", yref="paper",
            text=t,
            showarrow=False,
            font=dict(size=label_font_size, color=label_color),
            align="right",
            textangle=label_textangle,
            yanchor=yanchor,
            bgcolor="rgba(0,0,0,0)"   # без плашки
        )

    # если подписи над графиком — дадим место сверху
    if label_y != "inside":
        cur = fig.layout.margin or dict(l=10, r=10, t=50, b=10)
        fig.update_layout(margin=dict(l=cur["l"], r=cur["r"], b=cur["b"], t=max(cur["t"], 80)))
    return fig


def plot_gantt(
    df,
    start_column="Start",
    end_column="Finish",
    y_group="Y_Group",
    color_column="Resource",
    text_column="Task",
    graph_title="График занятости",
    font_size=12,
    bar_mode=False,
    sprint_df = None,
    width_plot=1600,
    height_plot=900,
    fit_names=True,
    font_size_names=12,
    max_len=80,
    swap_text=False,
    theme: dict = DEFAULT_THEME,
    color_map_override: dict | None = None,
    template_name: str = "plotly_dark",
    sprint_label_mode: str = "above",      # "inside" | "above"
    sprint_label_color: str | None = None,
):
    gantt_df = df.copy()
    if bar_mode:
        gantt_df = assign_tracks(gantt_df)
        gantt_df["Y_Group_lane"] = gantt_df["Y_Group"] + " (lane " + (gantt_df["track"]+1).astype(str) + ")"
        gantt_df["number_lane"] = (gantt_df["track"]+1).astype(str)
        gantt_df.sort_values(["Y_Group","number_lane"], ascending=False, inplace=True)
        y_group = "Y_Group_lane"

    if swap_text:
        gantt_df["Task"] = gantt_df["Task"].apply(lambda s: wrap_text(s, max_len=max_len))
        gantt_df["Y_Group"] = gantt_df["Y_Group"].apply(lambda s: wrap_text(s, max_len=max_len))
        gantt_df["Resource"] = gantt_df["Resource"].apply(lambda s: wrap_text(s, max_len=max_len))

    # палитра
    resources = gantt_df[color_column].astype(str).fillna("").unique().tolist()
    color_map = build_color_map(resources, theme, override=color_map_override)

    # построение
    fig = px.timeline(
        gantt_df,
        x_start=start_column, x_end=end_column,
        y=y_group, color=color_column, text=text_column,
        title=graph_title,
        opacity=theme.get("bar_opacity", 0.95),
        template=template_name,
        color_discrete_map=color_map,
    )

    # фиксируем ЕДИНЫЙ цвет текста (Plotly не будет автоинвертировать)
    bar_text_color = theme.get("bar_text_color", "#FFFFFF")
    fig.update_traces(
        textposition="inside",
        insidetextanchor="start",
        textangle=0,
        insidetextfont=dict(size=font_size, color=bar_text_color),
        textfont=dict(size=font_size, color=bar_text_color),
        width=0.85,
        offsetgroup=0,
        cliponaxis=False,           # текст может выступать за бар
        # constraintext="none",

    ),

    # ось X (недели)
    x_min = pd.to_datetime(gantt_df["Start"]).min()
    first_monday = x_min - pd.to_timedelta(x_min.weekday(), unit="d")
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=theme.get("grid_color"),
        tickformat="%d.%m",
        tick0=first_monday.strftime("%Y-%m-%d"),
        dtick=7 * 24 * 60 * 60 * 1000,
        linecolor=theme.get("grid_color"),
        zeroline=False
    )

    # ось Y
    y_cats = gantt_df[y_group].unique().tolist()
    fig.update_yaxes(
        automargin=True,
        tickfont=dict(size=font_size, color=theme.get("text_color")),
        categoryorder="array",
        categoryarray=y_cats,
        autorange="reversed",
        color=theme.get("text_color"),
        gridcolor=theme.get("grid_color"),
        zeroline=False
    )

    # общий стиль/фон/легенда
    fig.update_layout(
        height=height_plot,
        barmode="overlay",
        paper_bgcolor=theme.get("bg_paper"),
        plot_bgcolor=theme.get("bg_plot"),
        font=dict(color=theme.get("text_color")),
        margin=dict(l=16, r=16, t=90, b=12),   # чуть больше сверху под заголовок/лейблы
        legend=dict(
            title=color_column,
            bgcolor=theme.get("legend_bg"),
            bordercolor="rgba(255,255,255,0.08)",
            borderwidth=1,
            font=dict(color=theme.get("text_color"))
        ),
        title=dict(font=dict(color=theme.get("text_color"), size=font_size + 4))
    )
    if not fit_names:
        fig.update_layout(uniformtext_minsize=font_size_names, uniformtext_mode="show")

    # --- СПРИНТЫ: только от первой задачи, лейблы видимые ---
    if sprint_df is not None and len(sprint_df) > 0:
        
        first_task_start = pd.to_datetime(gantt_df[start_column]).min()
        s = sprint_df.copy()
        s["Дата начала"] = pd.to_datetime(s["Дата начала"], errors="coerce")
        s["Дата окончания"] = pd.to_datetime(s["Дата окончания"], errors="coerce")
        s = s[s["Дата окончания"] >= first_task_start].dropna(subset=["Дата начала", "Дата окончания"])

        label_color = sprint_label_color or theme.get("text_color", "#E8EAED")
        sprint_fill = theme.get("sprint_fill", "rgba(120,200,255,0.18)")

        # если хотим над графиком — добавим воздуха
        if sprint_label_mode == "above":
            cur = fig.layout.margin
            fig.update_layout(margin=dict(l=cur.l, r=cur.r, b=cur.b, t=max(cur.t, int(font_size*4))))

        for _, spr in s.iterrows():
            s0 = spr["Дата начала"]; s1 = spr["Дата окончания"]
            num = spr.get("Номер спринта", "")
            x0 = max(s0, first_task_start); x1 = s1

            fig.add_vrect(x0=x0, x1=x1, fillcolor=sprint_fill, opacity=1.0, layer="below", line_width=0)

            text = f"Спринт {num}<br>{s0:%d.%m.%Y}–{s1:%d.%m.%Y}"
            if sprint_label_mode == "inside":
                # внутри области графика, у верхней кромки — не обрежется
                fig.add_annotation(
                    x=x0 + (x1 - x0)/2, y=1.0, xref="x", yref="paper",
                    text=text, showarrow=False, align="center",
                    yanchor="top",
                    font=dict(size=font_size, color=label_color),
                    bgcolor="rgba(0,0,0,0)", borderpad=2
                )
            else:
                # над графиком (нужен больший margin.t)
                fig.add_annotation(
                    x=x0 + (x1 - x0)/2, y=1.02, xref="x", yref="paper",
                    text=text, showarrow=False, align="center",
                    yanchor="bottom",
                    font=dict(size=font_size, color=label_color),
                )

    # зебра по дорожкам
    z0, z1 = theme.get("zebra_colors", ("rgba(255,255,255,0.05)", "rgba(255,255,255,0.00)"))
    x_min = gantt_df["Start"].min(); x_max = gantt_df["Finish"].max()
    for idx, _cat in enumerate(y_cats):
        y0 = idx - 0.4; y1 = idx + 0.4
        fig.add_shape(
            type="rect", x0=x_min, x1=x_max, y0=y0, y1=y1,
            xref="x", yref="y",
            fillcolor=z0 if idx % 2 == 0 else z1,
            opacity=1.0, layer="below", line_width=0
        )

    markers = [
        {"date": "2025-11-17", "text": "Ветка на ПРОД"},
        {"date": "2025-11-28", "text": "Ветка на ЮАТ"},
        {"date": "2025-12-12", "text": "Закрытие заказ-наряда Q4/J,Обновление ПРОД"},
        {"date": "2025-12-26", "text": "Ветка на ПРОД"},
        ]
    fig = add_date_markers(
        fig, markers,
        line_color="#FF4D4D", line_width=2, line_dash="dash",
        label_color="#FF4D4D", label_font_size=11,
        label_y="inside",        # или "above"
        label_textangle= 90      # вертикальная подпись
    )
    # вывод
    chart_html = fig.to_html(include_plotlyjs='cdn', full_html=False)
    components_html.html(
        f"""
        <div style='width:100%; overflow-x:auto; background:#0B0D12;'>
          <div style='min-width:{width_plot}px;'>
            {chart_html}
          </div>
        </div>
        """,
        height=height_plot
    )
