import sqlite3

def save_memory(key, value):

    conn = sqlite3.connect("memory.db")

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT OR REPLACE INTO memory
        VALUES (?,?)
        """,
        (key, value)
    )

    conn.commit()
    conn.close()


def get_memory(key):

    conn = sqlite3.connect("memory.db")

    cursor = conn.cursor()

    cursor.execute(
        "SELECT value FROM memory WHERE key=?",
        (key,)
    )

    result = cursor.fetchone()

    conn.close()

    if result:
        return result[0]

    return None