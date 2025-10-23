import sqlite3
import datetime

DB_NAME = 'chat.db'

def get_db_connection():
    """获取一个数据库连接，并设置为字典返回模式"""
    conn = sqlite3.connect(DB_NAME)
    # 这行让查询结果可以像字典一样通过列名访问，非常方便
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """初始化数据库，创建表（如果它们还不存在的话）"""
    
    # SQL 命令来创建 'conversations' 表
    create_conversations_table_sql = """
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    """
    
    # SQL 命令来创建 'messages' 表
    # 'conversation_id' 是一个外键，它关联到 'conversations' 表的 'id'
    create_messages_table_sql = """
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id INTEGER NOT NULL,
        role TEXT NOT NULL,          -- 'user' 或 'assistant'
        content TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (conversation_id) REFERENCES conversations (id)
    );
    """

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        print("正在创建 'conversations' 表...")
        cursor.execute(create_conversations_table_sql)
        
        print("正在创建 'messages' 表...")
        cursor.execute(create_messages_table_sql)
        
        conn.commit()
        print(f"数据库 '{DB_NAME}' 已成功初始化。")

    except sqlite3.Error as e:
        print(f"数据库初始化时发生错误: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    # 当我们直接运行 `python database.py` 时，
    # 这部分代码会被执行。
    init_db()