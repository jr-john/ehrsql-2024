import re
import sqlite3
import multiprocessing as mp
from ast import literal_eval

__current_time = "2100-12-31 23:59:00"
__precomputed_dict = {
    "temperature": (35.5, 38.1),
    "sao2": (95.0, 100.0),
    "heart rate": (60.0, 100.0),
    "respiration": (12.0, 18.0),
    "systolic bp": (90.0, 120.0),
    "diastolic bp": (60.0, 90.0),
    "mean bp": (60.0, 110.0),
}


def post_process_sql(query):

    query = re.sub("[ ]+", " ", query.replace("\n", " ")).strip()
    query = query.replace("> =", ">=").replace("< =", "<=").replace("! =", "!=")

    if "current_time" in query:
        query = query.replace("current_time", f"'{__current_time}'")
    if re.search("[ \n]+([a-zA-Z0-9_]+_lower)", query) and re.search(
        "[ \n]+([a-zA-Z0-9_]+_upper)", query
    ):
        vital_lower_expr = re.findall("[ \n]+([a-zA-Z0-9_]+_lower)", query)[0]
        vital_upper_expr = re.findall("[ \n]+([a-zA-Z0-9_]+_upper)", query)[0]
        vital_name_list = list(
            set(
                re.findall("([a-zA-Z0-9_]+)_lower", vital_lower_expr)
                + re.findall("([a-zA-Z0-9_]+)_upper", vital_upper_expr)
            )
        )
        if len(vital_name_list) == 1:
            processed_vital_name = vital_name_list[0].replace("_", " ")
            if processed_vital_name in __precomputed_dict:
                vital_range = __precomputed_dict[processed_vital_name]
                query = query.replace(vital_lower_expr, f"{vital_range[0]}").replace(
                    vital_upper_expr, f"{vital_range[1]}"
                )

    query = query.replace("%y", "%Y").replace("%j", "%J")

    return query


def process_item(item):
    try:
        item = round(float(item), 3)
    except:
        pass
    return str(item)


def process_answer(ans):
    try:
        ans = literal_eval(ans)
    except:
        pass
    if type(ans) == str:
        return ans
    else:
        return str(
            sorted([[process_item(c) for c in row] for row in ans])[:100]
        )  # check only up to 100th record


def execute_sql(sql, db_path):
    con = sqlite3.connect(db_path)
    con.text_factory = lambda b: b.decode(errors="ignore")
    cur = con.cursor()
    result = cur.execute(sql).fetchall()
    con.close()
    return result


def execute_sql_wrapper(key, sql, db_path, tag, skip_indicator="null"):
    assert tag in ["real", "pred"]
    if sql != skip_indicator:
        try:
            result = execute_sql(sql, db_path)
        except:
            result = "error_" + tag
        result = process_answer(result)
        return (key, result)
    else:
        return (key, skip_indicator)


def execute_all(dict, db_path, tag):
    exec_result = {}
    for key in dict:
        sql = dict[key]
        exec_result[key] = execute_sql_wrapper(key, sql, db_path, tag)[-1]
    return exec_result


def execute_all_distributed(dict, db_path, tag, num_workers):
    exec_result = {}

    def result_tracker(result):
        exec_result[result[0]] = result[-1]

    pool = mp.Pool(processes=num_workers)
    for key in dict:
        sql = dict[key]
        pool.apply_async(
            execute_sql_wrapper, args=(key, sql, db_path, tag), callback=result_tracker
        )
    pool.close()
    pool.join()
    return exec_result
