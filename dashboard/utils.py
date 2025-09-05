import pandas as pd
import textwrap
import numpy as np



def working_hours_between(start_date, end_date, hours_per_day: int = 8) -> int:
    """
    Возвращает количество рабочих часов между двумя датами включительно.
    Учитываются только понедельник–пятница.
    
    Параметры:
    - start_date, end_date: строка 'YYYY-MM-DD' или datetime-подобный объект.
    - hours_per_day: число рабочих часов в одном дне (по умолчанию 8).
    """
    # Приводим к date
    start = pd.to_datetime(start_date).date()
    # Прибавляем 1 день к end_date, чтобы включить его в полуинтервал
    end_inclusive = pd.to_datetime(end_date).date() + pd.Timedelta(days=1)
    
    # Считаем рабочие дни в [start, end_inclusive)
    business_days = np.busday_count(
        start,
        end_inclusive,
        weekmask='Mon Tue Wed Thu Fri'
    )
    return business_days * hours_per_day

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

def wrap_text1(s, max_len=30, line_height="0.1em", letter_spacing="-0.3px"):
    """
    Переносит текст по max_len символов и оборачивает его в span с управлением
    межстрочным интервалом и расстоянием между буквами.

    Args:
        s (str|None): текст для переноса
        max_len (int): максимальная длина строки до переноса
        line_height (str): CSS line-height (например "0.9em" или "12px")
        letter_spacing (str): CSS letter-spacing (например "-0.5px")

    Returns:
        str: HTML-текст с <span> и <br>
    """
    if not s or not isinstance(s, str):
        return ""

    # перенос по max_len
    wrapped = "<br>".join(textwrap.wrap(s, width=max_len))

    # обёртка с CSS для управления плотностью текста
    styled = (
        f"<span style='line-height:{line_height}; "
        f"letter-spacing:{letter_spacing}; display:inline-block;'>"
        f"{wrapped}</span>"
    )
    return styled


def transform_backlog_to_summary(backlog_df,df_sprint,column_epic,column_task,mode=None):
        """
        Преобразует данные с листа "Бэклог" в формат "Сводной таблицы".
        Автоматически определяет роли по всем колонкам, кроме фиксированных.
        """
        print(df_sprint)
        # Динамически определяем роли — все остальные колонки
        roles = ['Эксперт RnD','Аналитик','Архитектор',"UX/UI","C#", "Py","React", "QA"]

        # Список для хранения строк в формате "Сводной таблицы"
        summary_rows = []

        for _, row in backlog_df.iterrows():
            for role in roles:
                # Добавляем строку только если есть информация по роли
                if not pd.isna(row[role]):
                    summary_rows.append({
                        "Feature (модуль)": row["Feature (модуль)"],
                        "Направление": row["Направление"],
                        "Название крупная задача (ЭПИК)": row[column_epic],
                        "Название задачи": row[column_task],
                        "Тип задачи": row["Тип задачи"],
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

        if mode == "Эпики":
            summary_df = (summary_df
                .groupby(["Название крупная задача (ЭПИК)", "Роль"], dropna=False)
                .agg({
                    "Feature (модуль)": lambda x: x.dropna().iloc[0] if not x.dropna().empty else None,
                    "Направление":      lambda x: x.dropna().iloc[0] if not x.dropna().empty else None,
                    "Исполнитель":      lambda x: x.mode().iloc[0] if not x.mode().empty else None,
                    "ТРЗ":              "sum",
                    "Номер спринта":    "min"
                })
                .reset_index()
)

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
    
import pandas as pd

def safe_calc_phases(df_g: pd.DataFrame, column_task_name: str = 'Задача') -> pd.DataFrame:
    """
    Безопасный вызов фазового расчёта (просто обёртка).
    """
    if df_g is None or df_g.empty:
        return df_g
    return calc_gantt_phases(df_g, column_task_name=column_task_name)

def transform_gantt(df):
    # 1. Преобразуем даты
    df["Start"] = pd.to_datetime(df["Start"])
    df["Finish"] = pd.to_datetime(df["Finish"])

    # 2. Группировка по Task
    agg_dict = {
        "Start": "min",
        "Finish": "max",
        "Task": lambda x: next((v for v in x if pd.notna(v) and v != "None"), None),
        "Resource": lambda x: next((v for v in x if pd.notna(v) and v != "None"), None),#lambda x: list(set(x.dropna()))
    }
    df_task = df.groupby("Y_Group", as_index=False).agg(agg_dict)

    # 3. Раскладываем задачи по дорожкам (чтобы не пересекались)
    tracks = []  # список дорожек, каждая дорожка = список (Finish последней задачи, список задач)
    placement = []  # результат

    for _, row in df_task.sort_values("Start").iterrows():
        placed = False
        for track_id, (last_finish, tasks) in enumerate(tracks):
            if row["Start"] >= last_finish:
                # кладём в эту дорожку
                tasks.append(row.to_dict())
                tracks[track_id] = (row["Finish"], tasks)
                placement.append({**row.to_dict(), "Track": track_id})
                placed = True
                break
        if not placed:
            # создаём новую дорожку
            track_id = len(tracks)
            tracks.append((row["Finish"], [row.to_dict()]))
            placement.append({**row.to_dict(), "Track": track_id})

    return pd.DataFrame(placement)