from src import db
from datetime import datetime

class Experiment(db.Model):
    __tablename__ = 'experiments'
    
    id = db.Column(db.Integer, primary_key=True)
    experiment_name = db.Column(db.String(200), nullable=False, comment='实验名称')
    virus_type = db.Column(db.String(50), comment='病毒类型')
    cell_line = db.Column(db.String(100), comment='细胞系')
    experiment_date = db.Column(db.Date, nullable=False, comment='实验日期')
    operator = db.Column(db.String(50), comment='操作人员')
    status = db.Column(db.String(20), default='pending', comment='实验状态')
    protocol = db.Column(db.Text, comment='实验方案')
    remarks = db.Column(db.Text, comment='备注')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    results = db.relationship('ExperimentResult', backref='experiment', lazy='dynamic')
    
    def to_dict(self):
        return {
            'id': self.id,
            'experiment_name': self.experiment_name,
            'virus_type': self.virus_type,
            'cell_line': self.cell_line,
            'experiment_date': self.experiment_date.isoformat() if self.experiment_date else None,
            'operator': self.operator,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class ExperimentResult(db.Model):
    __tablename__ = 'experiment_results'
    
    id = db.Column(db.Integer, primary_key=True)
    experiment_id = db.Column(db.Integer, db.ForeignKey('experiments.id'), nullable=False)
    concentration = db.Column(db.Float, comment='药物浓度(μM)')
    viability = db.Column(db.Float, comment='细胞存活率(%)')
    inhibition_rate = db.Column(db.Float, comment='抑制率(%)')
    ic50 = db.Column(db.Float, comment='IC50(μM)')
    cc50 = db.Column(db.Float, comment='CC50(μM)')
    si = db.Column(db.Float, comment='选择性指数(SI)')
    raw_data = db.Column(db.Text, comment='原始数据JSON')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'experiment_id': self.experiment_id,
            'concentration': self.concentration,
            'viability': self.viability,
            'inhibition_rate': self.inhibition_rate,
            'ic50': self.ic50,
            'cc50': self.cc50,
            'si': self.si
        }
