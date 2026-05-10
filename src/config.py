import os
from pathlib import Path
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / '.env'
load_dotenv(_ENV_FILE, override=False)

_DEFAULT_EXTERNAL_PDB = r'E:\graduationproject\分子对接+blast比对程序\蛋白存放'

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'flu-screen-secret-key-2024'

    DB_TYPE = os.environ.get('DB_TYPE', 'sqlite')

    if DB_TYPE == 'mysql':
        MYSQL_HOST = os.environ.get('MYSQL_HOST', 'localhost')
        MYSQL_PORT = int(os.environ.get('MYSQL_PORT', 3306))
        MYSQL_USER = os.environ.get('MYSQL_USER', 'root')
        MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD', '123123')
        MYSQL_DATABASE = os.environ.get('MYSQL_DATABASE', 'flu_screen')
        SQLALCHEMY_DATABASE_URI = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
    else:
        basedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        sqlite_path = os.path.join(basedir, 'data', 'flu_screen.db')
        os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
        SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', f'sqlite:///{sqlite_path}')

    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = False

    EXTERNAL_PDB_DIR = os.environ.get('EXTERNAL_PDB_DIR', '').strip()
    if not EXTERNAL_PDB_DIR:
        EXTERNAL_PDB_DIR = _DEFAULT_EXTERNAL_PDB