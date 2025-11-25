"""
AI-Based Phishing Detection System
Flask Application with ML Models
"""

from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import re
from urllib.parse import urlparse
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import traceback

# Flask App Configuration
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///phishing_detection.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

db = SQLAlchemy(app)

# ==================== DATABASE MODELS ====================

class PhishingReport(db.Model):
    __tablename__ = 'phishing_reports'
    id = db.Column(db.Integer, primary_key=True)
    email_subject = db.Column(db.String(255))
    sender_email = db.Column(db.String(255))
    message_content = db.Column(db.Text)
    urls = db.Column(db.JSON)
    classification = db.Column(db.String(20))
    confidence_score = db.Column(db.Float)
    detected_features = db.Column(db.JSON)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_feedback = db.Column(db.String(20))

    def to_dict(self):
        return {
            'id': self.id,
            'sender_email': self.sender_email,
            'subject': self.email_subject,
            'classification': self.classification,
            'confidence_score': round(self.confidence_score, 2),
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'detected_features': self.detected_features
        }

class UserAlert(db.Model):
    __tablename__ = 'user_alerts'
    id = db.Column(db.Integer, primary_key=True)
    alert_type = db.Column(db.String(50))
    message = db.Column(db.Text)
    severity = db.Column(db.String(20))
    report_id = db.Column(db.Integer, db.ForeignKey('phishing_reports.id'))
    is_read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class URLBlacklist(db.Model):
    __tablename__ = 'url_blacklist'
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(500), unique=True)
    threat_level = db.Column(db.String(20))
    added_date = db.Column(db.DateTime, default=datetime.utcnow)

# ==================== FEATURE EXTRACTION ====================

class PhishingDetector:
    def __init__(self):
        self.rf_model = None
        self.svm_model = None
        self.scaler = StandardScaler()
        self.load_trained_models()
        self.suspicious_keywords = [
            'verify', 'confirm', 'urgent', 'action required', 'click here',
            'update account', 'validate', 'suspended', 'limited time',
            'claim reward', 'winner', 'congratulations', 'unusual activity'
        ]

    def extract_url_features(self, urls):
        """Extract features from URLs"""
        features = {
            'has_suspicious_domain': 0,
            'url_length': 0,
            'has_ip_address': 0,
            'has_shortener': 0,
            'has_ssl': 0,
        }
        
        if not urls:
            return features
        
        shorteners = ['bit.ly', 'tinyurl', 'ow.ly', 'short.link']
        
        for url in urls:
            if url.strip():
                features['url_length'] += len(url)
                
                if re.match(r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url):
                    features['has_ip_address'] = 1
                
                if any(s in url.lower() for s in shorteners):
                    features['has_shortener'] = 1
                
                if 'https' in url:
                    features['has_ssl'] = 1
                
                domain = urlparse(url).netloc.lower()
                if len(domain) > 50 or '-' in domain or domain.count('.') > 2:
                    features['has_suspicious_domain'] = 1
        
        return features

    def extract_text_features(self, text, sender):
        """Extract features from email text and sender"""
        features = {
            'has_urgent_language': 0,
            'has_suspicious_keywords': 0,
            'suspicious_sender_format': 0,
            'text_length': len(text) if text else 0,
            'has_numbers_in_sender': 0,
        }
        
        text_lower = text.lower() if text else ''
        
        urgent_words = ['urgent', 'immediate', 'action required', 'verify now']
        if any(word in text_lower for word in urgent_words):
            features['has_urgent_language'] = 1
        
        keyword_count = sum(1 for kw in self.suspicious_keywords if kw in text_lower)
        if keyword_count > 2:
            features['has_suspicious_keywords'] = 1
        
        if sender:
            if re.search(r'\d', sender):
                features['has_numbers_in_sender'] = 1
            
            sender_domain = sender.split('@')[1].lower() if '@' in sender else ''
            if sender_domain and (len(sender_domain) > 30 or sender_domain.count('.') > 2):
                features['suspicious_sender_format'] = 1
        
        return features

    def extract_features_vector(self, email_data):
        """Create feature vector for ML models"""
        url_features = self.extract_url_features(email_data.get('urls', []))
        text_features = self.extract_text_features(
            email_data.get('content', ''),
            email_data.get('sender', '')
        )
        
        feature_vector = [
            url_features['has_suspicious_domain'],
            url_features['url_length'] / 100 if url_features['url_length'] > 0 else 0,
            url_features['has_ip_address'],
            url_features['has_shortener'],
            url_features['has_ssl'],
            text_features['has_urgent_language'],
            text_features['has_suspicious_keywords'],
            text_features['suspicious_sender_format'],
            text_features['text_length'] / 1000 if text_features['text_length'] > 0 else 0,
            text_features['has_numbers_in_sender']
        ]
        
        return np.array(feature_vector).reshape(1, -1), {**url_features, **text_features}

    def analyze_sender(self, sender):
        """Analyze sender email for suspicious patterns"""
        issues = []
        if '@' not in sender:
            issues.append('Invalid email format')
        else:
            domain = sender.split('@')[1].lower()
            if domain in ['gmail.com', 'yahoo.com', 'outlook.com']:
                issues.append('Free email service domain')
            if len(domain) > 50:
                issues.append('Unusually long domain')
            if '--' in domain or domain.count('-') > 3:
                issues.append('Suspicious domain pattern')
        
        return issues

    def predict(self, email_data):
        """Predict if email is phishing"""
        features_vector, detected_features = self.extract_features_vector(email_data)
        
        # If trained models are loaded, use them
        if self.rf_model is not None and self.svm_model is not None:
            try:
                features_scaled = self.scaler.transform(features_vector)
                
                rf_pred = self.rf_model.predict_proba(features_scaled)[0]
                svm_pred = self.svm_model.predict_proba(features_scaled)[0]
                
                # Average probability (0-1 range)
                phishing_confidence = (rf_pred[1] + svm_pred[1]) / 2
            except:
                # Fallback to heuristic if models fail
                # Count detected features and normalize to 0-1
                detected_count = sum(1 for v in detected_features.values() if v > 0)
                total_features = len(detected_features)
                phishing_confidence = detected_count / total_features
        else:
            # Use heuristic scoring
            # Count detected features and normalize to 0-1
            detected_count = sum(1 for v in detected_features.values() if v > 0)
            total_features = len(detected_features)
            phishing_confidence = detected_count / total_features
        
        # Ensure confidence is between 0 and 1
        phishing_confidence = min(max(float(phishing_confidence), 0.0), 1.0)
        
        threshold = 0.5
        classification = 'phishing' if phishing_confidence > threshold else 'legitimate'
        
        return {
            'classification': classification,
            'confidence_score': phishing_confidence,
            'detected_features': detected_features,
            'sender_issues': self.analyze_sender(email_data.get('sender', ''))
        }

    def load_trained_models(self):
        """Load pre-trained models if available"""
        import pickle
        try:
            # Check if model files exist and are not empty
            import os
            model_path = 'models/rf_model.pkl'
            if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
                with open(model_path, 'rb') as f:
                    self.rf_model = pickle.load(f)
                with open('models/svm_model.pkl', 'rb') as f:
                    self.svm_model = pickle.load(f)
                with open('models/scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
                print("✓ Trained models loaded successfully!")
            else:
                print("⚠ Model files are empty or corrupted.")
                print("  Run 'python train_models.py' to train models with Kaggle data.")
        except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
            print(f"⚠ Could not load models: {e}")
            print("  Using heuristic detection instead.")
            print("  Run 'python train_models.py' to train models with Kaggle data.")

# ==================== INITIALIZE DETECTOR ====================

detector = PhishingDetector()

# ==================== ROUTES ====================

@app.route('/')
def dashboard():
    """Dashboard home page"""
    try:
        return render_template('dashboard.html')
    except Exception as e:
        return f"Error loading dashboard: {str(e)}", 500

@app.route('/api/analyze', methods=['POST'])
def analyze_email():
    """Analyze email for phishing"""
    try:
        data = request.get_json()
        
        email_data = {
            'subject': data.get('subject', ''),
            'sender': data.get('sender', ''),
            'content': data.get('content', ''),
            'urls': data.get('urls', [])
        }
        
        result = detector.predict(email_data)
        
        report = PhishingReport(
            email_subject=email_data['subject'],
            sender_email=email_data['sender'],
            message_content=email_data['content'],
            urls=email_data['urls'],
            classification=result['classification'],
            confidence_score=result['confidence_score'],
            detected_features=result['detected_features']
        )
        
        db.session.add(report)
        
        if result['classification'] == 'phishing':
            alert = UserAlert(
                alert_type='Phishing Detected',
                message=f"Potential phishing email from {email_data['sender']}",
                severity='high',
                report_id=report.id
            )
            db.session.add(alert)
        
        db.session.commit()
        
        return jsonify({
            'report_id': report.id,
            'classification': result['classification'],
            'confidence': result['confidence_score'],
            'detected_features': result['detected_features'],
            'sender_issues': result['sender_issues']
        })
    except Exception as e:
        print(f"Error in analyze_email: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/reports', methods=['GET'])
def get_reports():
    """Get all phishing reports"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        
        reports = PhishingReport.query.order_by(PhishingReport.timestamp.desc()).paginate(
            page=page, per_page=per_page
        )
        
        return jsonify({
            'total': reports.total,
            'pages': reports.pages,
            'current_page': page,
            'reports': [r.to_dict() for r in reports.items]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Get unread alerts"""
    try:
        alerts = UserAlert.query.filter_by(is_read=False).order_by(
            UserAlert.created_at.desc()
        ).limit(50).all()
        
        return jsonify({
            'alerts': [{
                'id': a.id,
                'type': a.alert_type,
                'message': a.message,
                'severity': a.severity,
                'created_at': a.created_at.strftime('%Y-%m-%d %H:%M:%S')
            } for a in alerts]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get detection statistics"""
    try:
        total = PhishingReport.query.count()
        phishing = PhishingReport.query.filter_by(classification='phishing').count()
        legitimate = PhishingReport.query.filter_by(classification='legitimate').count()
        
        accuracy = (PhishingReport.query.filter(
            PhishingReport.user_feedback == 'correct'
        ).count() / max(1, total)) * 100
        
        return jsonify({
            'total_analyzed': total,
            'phishing_detected': phishing,
            'legitimate': legitimate,
            'accuracy_percentage': round(accuracy, 2),
            'phishing_percentage': round((phishing / max(1, total)) * 100, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print("✓ Database initialized")
        print("✓ Starting Flask app...")
        print("✓ Open your browser: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
    
