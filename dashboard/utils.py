import pandas as pd
import textwrap


# 1. Функция для назначения треков внутри каждой группы
def assign_tracks(df, group_col='Y_Group', start_col='Start', end_col='Finish'):
    df = df.sort_values([group_col, start_col]).reset_index(drop=True)
    tracks = {}           # для каждой группы — список окончания последнего бара в каждом треке
    track_ids = []        # сюда будем собирать номер трека для каждой строки
    for idx, row in df.iterrows():
        grp = row[group_col]
        st, fin = row[start_col], row[end_col]
        if grp not in tracks:
            tracks[grp] = []
        placed = False
        # пытаемся прицепить бар к самому раннему освободившемуся треку
        for i, last_end in enumerate(tracks[grp]):
            if st >= last_end:
                tracks[grp][i] = fin
                track_ids.append(i)
                placed = True
                break
        # если не сработало — открываем новый трек
        if not placed:
            tracks[grp].append(fin)
            track_ids.append(len(tracks[grp]) - 1)
    df['track'] = track_ids
    return df
def wrap_text(s, max_len=20):
    """
    Разбивает строку s на строки длиной до max_len символов,
    вставляя <br> для переноса в Plotly.
    """
    # textwrap.wrap разбивает по словам, но можно указать break_long_words=True
    lines = textwrap.wrap(str(s), width=max_len, break_long_words=True, replace_whitespace=False)
    return '<br>'.join(lines)


def transform_backlog_to_summary(backlog_df,df_sprint,column_name):
        """
        Преобразует данные с листа "Бэклог" в формат "Сводной таблицы".
        Автоматически определяет роли по всем колонкам, кроме фиксированных.
        """
        print(df_sprint)
        # Динамически определяем роли — все остальные колонки
        roles = ['Аналитик',"UX/UI","C#", "Py","React", "QA"]

        # Список для хранения строк в формате "Сводной таблицы"
        summary_rows = []

        for _, row in backlog_df.iterrows():
            for role in roles:
                # Добавляем строку только если есть информация по роли
                if not pd.isna(row[role]):
                    summary_rows.append({
                        "Процесс (модуль)": row["Процесс (модуль)"],
                        "Направление": row["Направление"],
                        "Название крупная задача (ЭПИК)": row[column_name],
                        "Роль": role,
                        "Исполнитель": row[role],
                        "ТРЗ": row["трз "+role],
                        "Номер спринта" : row["Номер спринта"],
                        # "Дата начала": pd.to_datetime('today').normalize(),
                        # "Дата конца": pd.to_datetime('today').normalize() + pd.to_timedelta(5, unit="d")
                        # "Дата конца": pd.to_datetime(row["Дата начала"]) + pd.to_timedelta(row["Продолжительность"], unit="d")
                    })

        # Создаём DataFrame из собранных данных
        summary_df = pd.DataFrame(summary_rows)

        # Преобразуем дату окончания в строку для удобства отображения
        # summary_df["Дата начала"] = summary_df["Дата начала"].dt.strftime("%Y-%m-%d")
        # summary_df["Дата конца"] = summary_df["Дата конца"].dt.strftime("%Y-%m-%d")

        return summary_df



def create_pivot(data, index, columns, values, aggfunc):
    try:
        return pd.pivot_table(
            data,
            index=index if index else None,
            columns=columns if columns else None,
            values=values,
            aggfunc=aggfunc,
            dropna=False
        )
    except Exception as e:
        st.error(f"Ошибка при создании сводной таблицы: {e}")
        return None
    

# Потестить гант диаграмму

# num_tasks = len(gantt_df)
# row_height_px = 30
# # Создаём Gantt-диаграмму
# fig1 = ff.create_gantt(
#     gantt_df,
#     index_col='Resource',    # по этому столбцу будет раскраска
#     show_colorbar=True,
#     group_tasks=False,       # ключевая опция — не наслаивать задачи
#     showgrid_x=True,
#     showgrid_y=True,
#     title="График занятости",
#     # height=2000,
#     bar_width=0.5,
#     height=num_tasks * row_height_px
# )

# chart_html = fig1.to_html(include_plotlyjs='cdn', full_html=False)
# # embed using components_html
# components_html.html(
#     f"""
#     <div style='width:100%; overflow-x:auto;'>
#       <div style='min-width:2000px;'>
#         {chart_html}
#       </div>
#     </div>
#     """,
#     height=1020
# )
# st.plotly_chart(fig1, use_container_width=True)