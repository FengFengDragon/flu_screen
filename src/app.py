import os
import sys

BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from flask import Flask, render_template
from flask_cors import CORS
from flask_migrate import Migrate
from src.config import Config
from src import db

def create_app(config_class=Config):
    app = Flask(__name__, 
                template_folder=os.path.join(BASE_DIR, 'templates'),
                static_folder=os.path.join(BASE_DIR, 'static'))
    app.config.from_object(config_class)

    db.init_app(app)
    CORS(app)
    Migrate(app, db)

    from src.routes import virtual_screening
    app.register_blueprint(virtual_screening.bp)

    from src.routes import ml_binding
    app.register_blueprint(ml_binding.bp)

    from src.routes import trajectory_analysis
    app.register_blueprint(trajectory_analysis.bp)

    from src.routes import trajectory_visualization
    app.register_blueprint(trajectory_visualization.bp)

    from src.routes import molecular_dynamics
    app.register_blueprint(molecular_dynamics.bp)

    from src.routes import workflow
    app.register_blueprint(workflow.bp)

    enable_deep_learning = os.environ.get('ENABLE_DEEP_LEARNING', '0') == '1' or app.config.get('ENABLE_DEEP_LEARNING', False)
    if enable_deep_learning:
        from src.routes import deep_learning
        app.register_blueprint(deep_learning.bp)
        print("[OK] 深度学习模块已启用")
    else:
        print("[INFO] 深度学习模块已禁用（设置环境变量 ENABLE_DEEP_LEARNING=1 启用）")

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/virtual-screening')
    def virtual_screening_page():
        return render_template('virtual_screening.html')

    @app.route('/ml-binding')
    def ml_binding_page():
        return render_template('ml_binding.html')

    @app.route('/deep-learning')
    def deep_learning_page():
        return render_template('deep_learning.html')

    @app.route('/trajectory-analysis')
    def trajectory_analysis_page():
        return render_template('trajectory_analysis.html')

    @app.route('/trajectory-visualization')
    def trajectory_visualization_page():
        return render_template('trajectory_visualization.html')

    @app.route('/md-simulation')
    def md_simulation_page():
        return render_template('md_simulation.html')

    @app.route('/workflow')
    def workflow_page():
        return render_template('workflow.html')

    @app.route('/health')
    def health():
        return {'status': 'healthy'}

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)
