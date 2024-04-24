import fire
import contextlib
import pandas as pd


def load_schema(DATASET_JSON):
    """This function loads and processes a database schema from a JSON file."""
    schema_df = pd.read_json(DATASET_JSON)
    schema_df = schema_df.drop(["column_names", "table_names"], axis=1)
    schema = []
    f_keys = []
    p_keys = []
    for index, row in schema_df.iterrows():
        tables = row["table_names_original"]
        col_names = row["column_names_original"]
        col_types = row["column_types"]
        foreign_keys = row["foreign_keys"]
        primary_keys = row["primary_keys"]
        for col, col_type in zip(col_names, col_types):
            index, col_name = col
            if index > -1:
                schema.append([row["db_id"], tables[index], col_name, col_type])
        for primary_key in primary_keys:
            index, column = col_names[primary_key]
            p_keys.append([row["db_id"], tables[index], column])
        for foreign_key in foreign_keys:
            first, second = foreign_key
            first_index, first_column = col_names[first]
            second_index, second_column = col_names[second]
            f_keys.append(
                [
                    row["db_id"],
                    tables[first_index],
                    tables[second_index],
                    first_column,
                    second_column,
                ]
            )
    db_schema = pd.DataFrame(
        schema, columns=["Database name", "Table Name", "Field Name", "Type"]
    )
    primary_key = pd.DataFrame(
        p_keys, columns=["Database name", "Table Name", "Primary Key"]
    )
    foreign_key = pd.DataFrame(
        f_keys,
        columns=[
            "Database name",
            "First Table Name",
            "Second Table Name",
            "First Table Foreign Key",
            "Second Table Foreign Key",
        ],
    )
    return db_schema, primary_key, foreign_key


def construct_tree(schema, pks, fks):
    """This function constructs a tree representation of the database schema."""
    tree = {}

    for index, row in schema.iterrows():
        db = row["Database name"]
        table = row["Table Name"]
        field = row["Field Name"]
        dtype = row["Type"]

        if db not in tree:
            tree[db] = {}

        if table not in tree[db]:
            pk = (
                pks.where(pks["Table Name"] == "patients")
                .dropna()["Primary Key"]
                .item()
            )
            fk_list = [
                {
                    "from_table": row["First Table Name"],
                    "from_key": row["First Table Foreign Key"],
                    "key": row["Second Table Foreign Key"],
                }
                for item, row in fks.where(fks["Second Table Name"] == table)
                .dropna()
                .iterrows()
            ]

            tree[db][table] = {"fields": [], "pk": pk, "fks": fk_list}

        if field not in tree[db][table]["fields"]:
            tree[db][table]["fields"].append({"name": field, "type": dtype})

    return tree


def print_mysql_ddl(tree):
    """This function prints the MySQL DDL commands for creating the database schema."""
    typemap = {
        "number": "INTEGER",
        "text": "TEXT",
        "time": "TIME",
    }

    def create_db(db):
        return f"CREATE DATABASE {db};"

    def use_db(db):
        return f"USE {db};"

    def create_table(table, fields, pk, fks):
        """Create a table in the database schema."""
        field_data = ",\n    ".join(
            [f"{field['name']} {typemap[field['type']]}" for field in fields]
        )
        fk_data = ",\n    ".join(
            [
                f"FOREIGN KEY ({fk['from_key']}) REFERENCES {fk['from_table']}({fk['key']})"
                for fk in fks
            ]
        )
        return f"""CREATE TABLE {table} (
    {field_data},
    PRIMARY KEY ({pk}),
    {fk_data}
);
"""

    def recursively_create_tables(tree, table, created_tables=set()):
        """Recursively create tables in the database schema."""
        if table in created_tables:
            return

        for fk in tree[table]["fks"]:
            recursively_create_tables(tree, fk["from_table"], created_tables)

        print(
            create_table(
                table, tree[table]["fields"], tree[table]["pk"], tree[table]["fks"]
            )
        )
        created_tables.add(table)

    for db in tree:
        # print(create_db(db))
        # print(use_db(db))

        for table in tree[db]:
            recursively_create_tables(tree[db], table)


def main(
    output_path="schema_prompt.txt",  # path to save the output
    ehrsql_path="../../ehrsql-2024",  # path to the EHRSQL git repository
):
    tables_path = f"{ehrsql_path}/data/mimic_iv/tables.json"
    schema, pks, fks = load_schema(tables_path)
    tree = construct_tree(schema, pks, fks)

    with open(output_path, "w") as f:
        with contextlib.redirect_stdout(f):
            print_mysql_ddl(tree)


if __name__ == "__main__":
    fire.Fire(main)
