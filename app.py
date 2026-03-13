from flask import Flask, request, jsonify, redirect, Response
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from pymongo import MongoClient
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re
from dotenv import load_dotenv
from werkzeug.middleware.proxy_fix import ProxyFix

from auth.auth_utils import hash_password, verify_password
from model import SentimentModel

load_dotenv()

model_error_message = ""
sentiment_model = None


def _ensure_model_ready():
    global sentiment_model, model_error_message
    if sentiment_model is not None:
        return True
    try:
        sentiment_model = SentimentModel(
            api_token=os.getenv("HUGGINGFACE_API_TOKEN") or os.getenv("HF_API_TOKEN")
        )
        model_error_message = ""
        return True
    except Exception as e:
        model_error_message = str(e)
        print(f"Warning: Sentiment model initialization failed: {e}")
        return False

app = Flask(__name__)
CORS(app)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-in-production')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=30)
app.config['PREFERRED_URL_SCHEME'] = 'https'
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

jwt = JWTManager(app)
DEBUG_MODE = os.getenv('FLASK_DEBUG', 'false').lower() in {'1', 'true', 'yes', 'on'}


def _internal_server_error(log_message):
    app.logger.exception(log_message)
    return jsonify({'error': 'Internal server error'}), 500


@app.before_request
def enforce_https():
    if app.debug:
        return None
    if request.is_secure or request.headers.get('X-Forwarded-Proto', 'http') == 'https':
        return None
    if request.host.startswith('localhost') or request.host.startswith('127.0.0.1'):
        return None
    return redirect(request.url.replace('http://', 'https://', 1), code=301)


MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
client = MongoClient(MONGO_URI)
db = client['sarcasm_detection']

users_collection = db['users']
predictions_collection = db['predictions']
feedback_collection = db['feedback']

users_collection.create_index('email', unique=True)
users_collection.create_index('username', unique=True)


def _context_response(text, reason, message):
    return jsonify({
        'text': text,
        'label': 'unknown',
        'is_sarcasm': None,
        'conclusion': 'Unable to conclude sarcasm from the given input.',
        'needs_more_context': True,
        'reason': reason,
        'message': message,
        'created_at': datetime.utcnow().isoformat()
    }), 200


def _is_probable_greeting_or_smalltalk(text):
    lowered = str(text or "").lower().strip()
    cleaned = re.sub(r'[^a-z\s]', ' ', lowered)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    words = cleaned.split() if cleaned else []

    if len(words) > 7:
        return False

    return is_greeting(lowered)


def is_greeting(text):
    cleaned = str(text or "").lower().strip()

    greeting_patterns = [
        r'^(hi|hello|hey|hii+|yo)\b',
        r'^(good\s+(morning|afternoon|evening|night))\b',
        r'^(how\s+are\s+you|how\s+r\s+u|hows\s+it\s+going|how\s+is\s+it\s+going)\b',
        r'^(thanks|thank\s+you|thank\s+u)$',
        r'^(whats\s+up|what\s+is\s+up|sup)\b',
    ]

    return any(re.match(pattern, cleaned) for pattern in greeting_patterns)


# ===============================
# Context Check
# ===============================

def needs_more_context(text, min_words=2):
    if not isinstance(text, str):
        return True
    words = re.findall(r"\b\w+\b", text.strip())
    return len(words) < min_words


def _normalize_for_patterns(text):
    cleaned = str(text or "").lower()
    replacements = {
        "’": "'",
        "`": "'",
        "“": '"',
        "”": '"',
        "…": "...",
    }
    for source, target in replacements.items():
        cleaned = cleaned.replace(source, target)
    cleaned = cleaned.replace("\u2019", "'").replace("\u2018", "'")
    cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"')
    cleaned = cleaned.replace("\u2026", "...")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# ===============================
# Strong Explicit Sarcasm Signals
# ===============================

def _has_strong_sarcasm_signal(text):
    lowered = _normalize_for_patterns(text)
    strong_patterns = [
        r'\byeah right\b',
        r'\bas if\b',
        r'\bsure jan\b',
        r'\boh great\b',
        r'\boh fantastic\b',
        r'\boh wonderful\b',
        r'\blove that for me\b',
        r'\bwhat a surprise\b',
        r'\bhow original\b',
        r'\bjust perfect\b',
        r'\bexactly what i needed\b',
        r'\bjust what i wanted\b',
        r'\bsure,\s*because\b',
        r'\byeah,\s*because\b',
        r'\bnothing says \w+ like\b',
        r'\bwhat could go wrong\b',
        r'\bgreat timing\b',
        r'\bmy favorite hobby\b',
        r'\bexactly the kind of surprise i enjoy\b',
        r'\bthis is fine\b',
        r'\bbrilliant\b',
        r'\bperfect timing\b',
        r'\bjust amazing\b',
        r'\bhow nice\b',
        r'\bwhat a lovely\b',
        r'\bthat went well\b',
        r'\bthat worked out great\b',
        r'/s\b',
        r'!{2,}',
        r'\?{2,}',
        r'\.\.\.',
        r'\bjust what i needed\b',
        r'\bwhat could possibly go wrong\b',
        r'\byeah because\b',
        r'\bthanks for nothing\b',
        r'\bwell that went well\b',
        r'\bthis day keeps getting better\b',
        r'\?!',
        r'\bthanks a lot\b',
        r'\byeah sure\b',
        r'\bcould this day get any better\b',
        r'\bexactly how i wanted\b',
        r'\bcould this (day|week|moment) get any better\b',
        r'\boh nice\b',
        r'\bjust great\b',
        r'\bnice timing\b',
        r'\bgreat another\b',
        r'\bi love when\b',
        r'\bi love how\b',
        r'\bi just love when\b',
    ]
    return any(re.search(pattern, lowered) for pattern in strong_patterns)


# ===============================
# Positive + Negative Mismatch
# ===============================

def _has_ironic_positive_negative_mismatch(text):
    text = _normalize_for_patterns(text)
    positive_openers = [
        r"\bi (just )?(love|adore|enjoy)\b",
        r"\bi (really )?enjoy(ed)?\b",
        r"\bi (totally|absolutely|really) (love|enjoy)\b",
        r"\bi'?m so excited\b",
        r"\bi'?m thrilled\b",
        r"\bso (great|amazing|wonderful|fun|fantastic)\b",
        r"\bwhat a (great|wonderful|perfect|fantastic)\b",
        r"\b(wonderful|perfect|great|fantastic|amazing),\b",
        r"\bbest (part|thing)\b",
        r"\bmy favorite hobby\b",
        r"\bso happy\b",
        r"\bso glad\b",
    ]
    negative_context = [
        r"\btraffic\b",
        r"\bdelay(ed|s)?\b",
        r"\bwaiting\b",
        r"\blate\b",
        r"\bstuck\b",
        r"\bbroken\b",
        r"\berror(s)?\b",
        r"\bcrash(ed|es)?\b",
        r"\bbug(s)?\b",
        r"\bdeadline\b",
        r"\bovertime\b",
        r"\bexam\b",
        r"\bfail(ed|ure)?\b",
        r"\bslow\b",
        r"\bforgot\b",
        r"\bassignment(s)?\b",
        r"\bclean(ing)?\b",
        r"\bhouse\b",
        r"\brain(ing)?\b",
        r"\bgroup project\b",
        r"\blegacy code\b",
        r"\bdebug(ging)?\b",
        r"\brejection\b",
        r"\bemail\b",
        r"\bmeeting\b",
        r"\bpower cut\b",
        r"\bnetwork issue\b",
        r"\bupdate\b",
        r"\bserver down\b",
        r"\btimeout\b",
        r"\blost\b",
        r"\bpower\b.*\b(out|cut)\b",
        r"\bdelete(d|ing)?\b",
        r"\bnot working\b",
        r"\bfroze\b",
        r"\bshutdown\b",
        r"\binternet (down|gone)\b",
        r"\bcleaning\b",
        r"\bchores\b",
        r"\bhousework\b",
        r"\btraffic jam\b",
r"\bpower outage\b",
r"\bbattery dead\b",
r"\bphone died\b",
r"\bslow internet\b",
r"\bmeeting\b",
r"\boffice\b",
r"\bwork\b",
r"\bdeadline\b",
r"\bqueue\b",
r"\bline\b",
r"\bflight delay\b",
    ]

    has_positive = any(re.search(p, text) for p in positive_openers)
    has_negative = any(re.search(n, text) for n in negative_context)

    return has_positive and has_negative


# ===============================
# Additional Sarcasm Patterns
# ===============================

def _has_additional_patterns(text):
    text = _normalize_for_patterns(text)

    additional_patterns = [
        r"\bwow[,! ]",
        r"\bamazing[,! ]",
        r"\bfantastic[,! ]",
        r"\bbrilliant[,! ]",
        r"\bgreat[,! ]+another\b",
        r"\bnice[,! ]",
        r"\bwonderful[,! ]",
        r"\bgreat job breaking\b",
        r"\bnice work breaking\b",
        r"\bgood job breaking\b",
        r"\bthat worked well\b",
        r"\bthat went well\b",
        r"\bcould this (day|week|get) any better\b",
        r"\bisn't that just great\b",
        r"\bwho wouldn'?t want that\b",
        r"\bwhat else could go wrong\b",
        r"\bhow could this possibly fail\b",
        r"\bare you serious\b",
        r"\bare you kidding\b",
        r"\bseriously\b",
        r"\breally now\b",
        r"\bwhat a surprise\b",
        r"\bmust be nice\b",
        r"\bgood for you\b",
        r"\bi'm thrilled\b",
        r"\bi'm so excited\b",
        r"\bwell good for you\b",
        r"\blove that\b",
        r"\bsounds fun\b",
        r"\bperfect\b$",
        r"\bgreat\b$",
        r"\bnice\b$",
        r"\bthanks a lot\b",
        r"\bthanks for nothing\b",
        r"\bjust fantastic\b",
        r"\bthis is going great\b",
        r"\bthat escalated quickly\b",
        r"\bwhat a disaster\b",
        r"\banother software update\b",
        r"\bdebugging all night\b",
        r"\bserver crashed\b",
        r"\bcode broke\b",
        r"\bperfect,\s*it started raining\b",
        r"\bjust what i needed,\s*another\b",
        r"\bexactly what i needed,\s*more\b",
        r"\bthis day keeps getting better\b",
        r"\bmy day is complete\b",
    ]
    return any(re.search(pattern, text) for pattern in additional_patterns)


def _has_positive_then_negative_event(text):
    text = _normalize_for_patterns(text)
    positive_intro = re.search(
        r"^\s*(great|perfect|amazing|fantastic|wonderful|nice|brilliant)\b[,.!]?",
        text,
    )
    if not positive_intro:
        return False
    negative_event = re.search(
        r"\b(went out|stopped working|is down|crashed|failed|froze|broke|not working|got corrupted|died)\b",
        text,
    )
    return bool(negative_event)


def _is_context_dependent_praise(text):
    text = _normalize_for_patterns(text)
    patterns = [
        r"^\s*well done[.!]?$",
        r"^\s*nice work[.!]?$",
        r"^\s*good job[.!]?$",
        r"^\s*everything is perfect[.!]?$",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


# ===============================
# Literal Positive Protection
# ===============================

def _is_literal_positive(text):
    text = _normalize_for_patterns(text)
    literal_patterns = [
        r"\bi am happy\b",
        r"\bfeeling great\b",
        r"\bhad a good day\b",
        r"\bthank you\b$",
        r"\bthanks\b$",
        r"\bgood morning\b",
        r"\bhello\b",
        r"\beverything is fine\b",
        r"\bi appreciate it\b",
        r"\bi really like\b",
        r"\bthat sounds good\b",
        r"\bthat is great\b",
        r"\bi enjoyed it\b",
        r"\bcongratulations\b",
        r"\bwell done[.!]?$",
        r"\bnice work[.!]?$",
        r"\bgood job[.!]?$",
        r"\bhe did a good job\b",
        r"\bgreat job[.!]?$",
        r"\beverything is perfect[.!]?$",
    ]
    return any(re.search(pattern, text) for pattern in literal_patterns)


# ===============================
# Connectivity Failure Pattern
# ===============================

def _has_connectivity_sarcasm(text):
    text = _normalize_for_patterns(text)

    has_connectivity_issue = (
        re.search(r"\b(internet|wifi|wi-fi|network)\b", text)
        and re.search(r"\b(stop(s|ped)? working|not working|down|fail(s|ed)?|disconnect(ed|s)?)\b", text)
    )

    has_context = re.search(r"\b(meeting|meetings|call|calls|interview|interviews|class|classes)\b", text)

    return has_connectivity_issue and has_context


# ===============================
# Main Heuristic Scoring Function
# ===============================

def sarcasm_heuristic_score(text):
    lowered = _normalize_for_patterns(text)

    if not lowered:
        return 0.0

    if is_greeting(lowered):
        return 0.0

    if needs_more_context(lowered):
        return 0.1

    score = 0.0

    if _has_strong_sarcasm_signal(lowered):
        score = max(score, 0.90)

    if _has_ironic_positive_negative_mismatch(lowered):
        score = max(score, 0.85)

    if _has_connectivity_sarcasm(lowered):
        score = max(score, 0.80)

    if _has_positive_then_negative_event(lowered):
        score = max(score, 0.74)

    if _has_additional_patterns(lowered):
        score = max(score, 0.65)

    if _is_literal_positive(lowered):
        score = min(score, 0.2)

    if _is_context_dependent_praise(lowered) and not _has_ironic_positive_negative_mismatch(lowered):
        score = min(score, 0.2)

    return score


def _normalize_sentiment_label(label):
    normalized = str(label or "").strip().lower()
    aliases = {
        "label_0": "negative",
        "label_1": "neutral",
        "label_2": "positive",
        "neg": "negative",
        "neu": "neutral",
        "pos": "positive",
    }
    return aliases.get(normalized, normalized)


def _to_sarcasm_result(text, raw_predictions):
    scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

    for item in raw_predictions or []:
        if not isinstance(item, dict):
            continue
        sentiment = _normalize_sentiment_label(item.get("label"))
        if sentiment in scores:
            scores[sentiment] = float(item.get("score", 0.0))

    sentiment_sarcasm_score = scores["negative"]
    heuristic_sarcasm_score = sarcasm_heuristic_score(text)

    # Negative sentiment alone is not sarcasm. Use cues as primary signal.
    if heuristic_sarcasm_score >= 0.55:
        sarcastic_score = max(
            heuristic_sarcasm_score,
            min(1.0, (0.35 * sentiment_sarcasm_score) + (0.75 * heuristic_sarcasm_score)),
        )
    elif heuristic_sarcasm_score >= 0.35:
        sarcastic_score = max(
            heuristic_sarcasm_score,
            min(1.0, (0.40 * sentiment_sarcasm_score) + (0.60 * heuristic_sarcasm_score)),
        )
    else:
        sarcastic_score = min(0.35, 0.30 * sentiment_sarcasm_score)

    sarcastic_score = max(0.0, min(1.0, sarcastic_score))
    non_sarcastic_score = 1.0 - sarcastic_score
    is_sarcastic = sarcastic_score >= 0.42
    label = "sarcastic" if is_sarcastic else "not sarcastic"

    return {
        "label": label,
        "is_sarcasm": is_sarcastic,
        "sarcastic_percentage": round(sarcastic_score * 100, 2),
        "non_sarcastic_percentage": round(non_sarcastic_score * 100, 2),
    }


def _sarcasm_conclusion(is_sarcasm):
    return "The sentence is sarcastic." if is_sarcasm else "The sentence is not sarcastic."


@app.route('/api/signup', methods=['POST'])
def signup():
    try:
        data = request.get_json(silent=True) or {}

        if not data.get('username') or not data.get('email') or not data.get('password'):
            return jsonify({'error': 'Missing required fields'}), 400

        if users_collection.find_one({'email': data['email']}):
            return jsonify({'error': 'Email already registered'}), 400

        if users_collection.find_one({'username': data['username']}):
            return jsonify({'error': 'Username already taken'}), 400

        user = {
            'username': data['username'],
            'email': data['email'],
            'password': hash_password(data['password']),
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }

        result = users_collection.insert_one(user)

        return jsonify({
            'message': 'User registered successfully',
            'user_id': str(result.inserted_id)
        }), 201

    except Exception:
        return _internal_server_error('Signup failed')


@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json(silent=True) or {}

        if not data.get('email') or not data.get('password'):
            return jsonify({'error': 'Email and password required'}), 400

        user = users_collection.find_one({'email': data['email']})

        if not user or not verify_password(user['password'], data['password']):
            return jsonify({'error': 'Invalid email or password'}), 401

        access_token = create_access_token(identity=str(user['_id']))

        return jsonify({
            'access_token': access_token,
            'username': user['username'],
            'user_id': str(user['_id'])
        }), 200

    except Exception:
        return _internal_server_error('Login failed')


@app.route('/api/user/profile', methods=['GET'])
@jwt_required()
def get_profile():
    try:
        user_id = get_jwt_identity()
        from bson import ObjectId

        user = users_collection.find_one({'_id': ObjectId(user_id)})

        if not user:
            return jsonify({'error': 'User not found'}), 404

        return jsonify({
            'user_id': str(user['_id']),
            'username': user['username'],
            'email': user['email'],
            'created_at': user['created_at'].isoformat()
        }), 200

    except Exception:
        return _internal_server_error('Profile lookup failed')


@app.route('/api/predict', methods=['POST'])
@jwt_required()
def predict_sarcasm():
    try:
        user_id = get_jwt_identity()
        data = request.get_json(silent=True) or {}
        from bson import ObjectId

        if not data.get('text'):
            return jsonify({'error': 'Text is required'}), 400

        text = data['text']

        if needs_more_context(text, min_words=1) and not _has_strong_sarcasm_signal(text):
            return _context_response(
                text,
                'short_input',
                'Text is empty. Please enter a sentence for prediction.'
            )

        if not _ensure_model_ready():
            details = f' ({model_error_message})' if model_error_message else ''
            return jsonify({'error': f'Sentiment model not available{details}'}), 503

        try:
            result = sentiment_model.predict(text)
            all_scores = result.get('raw', [])
            sarcasm = _to_sarcasm_result(text, all_scores)
        except Exception as model_error:
            app.logger.exception(f'Model prediction failed for single input: {model_error}')
            return jsonify({'error': 'Model prediction failed'}), 500

        if _is_probable_greeting_or_smalltalk(text):
            return _context_response(
                text,
                'greeting_input',
                'This looks like a greeting/small-talk message. Please enter fuller text for reliable prediction.'
            )

        prediction = {
            'user_id': ObjectId(user_id),
            'text': text,
            'label': sarcasm['label'],
            'is_sarcasm': sarcasm['is_sarcasm'],
            'sarcastic_percentage': sarcasm['sarcastic_percentage'],
            'non_sarcastic_percentage': sarcasm['non_sarcastic_percentage'],
            'conclusion': _sarcasm_conclusion(sarcasm['is_sarcasm']),
            'created_at': datetime.utcnow()
        }

        result = predictions_collection.insert_one(prediction)

        return jsonify({
            'prediction_id': str(result.inserted_id),
            'text': text,
            'label': prediction['label'],
            'is_sarcasm': prediction['is_sarcasm'],
            'sarcastic_percentage': prediction['sarcastic_percentage'],
            'non_sarcastic_percentage': prediction['non_sarcastic_percentage'],
            'conclusion': prediction['conclusion'],
            'created_at': prediction['created_at'].isoformat()
        }), 200

    except Exception:
        return _internal_server_error('Single prediction failed')


@app.route('/api/predict/batch', methods=['POST'])
@jwt_required()
def predict_sarcasm_batch():
    try:
        data = request.get_json(silent=True) or {}
        texts = data.get('texts')
        try:
            max_items = int(data.get('max_items', 2000))
        except (TypeError, ValueError):
            return jsonify({'error': 'max_items must be an integer between 1 and 5000'}), 400

        if max_items < 1 or max_items > 5000:
            return jsonify({'error': 'max_items must be between 1 and 5000'}), 400

        if not isinstance(texts, list):
            return jsonify({'error': "Provide 'texts' as an array"}), 400
        if not texts:
            return jsonify({'error': 'No sentences provided'}), 400
        if len(texts) > max_items:
            return jsonify({'error': f'Input contains {len(texts)} sentences, exceeds max_items={max_items}'}), 400

        if not _ensure_model_ready():
            details = f' ({model_error_message})' if model_error_message else ''
            return jsonify({'error': f'Sentiment model not available{details}'}), 503

        def _predict_one(index, text):
            pred = sentiment_model.predict(text)
            sarcasm = _to_sarcasm_result(text, pred.get('raw', []))
            return {
                'index': index,
                'text': text,
                'label': sarcasm['label'],
                'is_sarcasm': sarcasm['is_sarcasm'],
                'sarcastic_percentage': sarcasm['sarcastic_percentage'],
                'non_sarcastic_percentage': sarcasm['non_sarcastic_percentage'],
                'conclusion': _sarcasm_conclusion(sarcasm['is_sarcasm'])
            }

        try:
            worker_limit = int(os.getenv('BATCH_PREDICT_WORKERS', '12'))
        except (TypeError, ValueError):
            worker_limit = 12
        max_workers = min(max(1, worker_limit), 32, len(texts))

        items = [None] * len(texts)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_predict_one, idx, text): idx
                for idx, text in enumerate(texts)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                text = texts[idx]
                try:
                    items[idx] = future.result()
                except Exception as pred_error:
                    app.logger.exception(f'Model prediction failed for batch index {idx}: {pred_error}')
                    items[idx] = {
                        'index': idx,
                        'text': text,
                        'error': 'Model prediction failed for this item'
                    }

        return jsonify({
            'count': len(items),
            'items': items
        }), 200

    except Exception:
        return _internal_server_error('Batch prediction failed')


@app.route('/api/history', methods=['GET'])
@jwt_required()
def get_history():
    try:
        user_id = get_jwt_identity()
        from bson import ObjectId

        page = request.args.get('page', 1, type=int)
        limit = request.args.get('limit', 10, type=int)

        skip = (page - 1) * limit

        predictions = list(predictions_collection.find(
            {'user_id': ObjectId(user_id)}
        ).sort('created_at', -1).skip(skip).limit(limit))

        for pred in predictions:
            pred['_id'] = str(pred['_id'])
            pred['user_id'] = str(pred['user_id'])
            pred['created_at'] = pred['created_at'].isoformat()
            pred['label'] = 'sarcastic' if pred.get('is_sarcasm') else 'not sarcastic'
            pred['conclusion'] = _sarcasm_conclusion(bool(pred.get('is_sarcasm')))
            if 'sarcastic_percentage' not in pred or 'non_sarcastic_percentage' not in pred:
                if pred.get('is_sarcasm') is True:
                    pred['sarcastic_percentage'] = 100.0
                    pred['non_sarcastic_percentage'] = 0.0
                elif pred.get('is_sarcasm') is False:
                    pred['sarcastic_percentage'] = 0.0
                    pred['non_sarcastic_percentage'] = 100.0
                else:
                    pred['sarcastic_percentage'] = None
                    pred['non_sarcastic_percentage'] = None
            pred.pop('confidence', None)
            pred.pop('raw', None)
            pred.pop('sentiment_scores', None)

        total_count = predictions_collection.count_documents({'user_id': ObjectId(user_id)})

        return jsonify({
            'predictions': predictions,
            'total': total_count,
            'page': page,
            'limit': limit
        }), 200

    except Exception:
        return _internal_server_error('History fetch failed')


@app.route('/api/prediction/<prediction_id>', methods=['GET'])
@jwt_required()
def get_prediction(prediction_id):
    try:
        user_id = get_jwt_identity()
        from bson import ObjectId

        prediction = predictions_collection.find_one({
            '_id': ObjectId(prediction_id),
            'user_id': ObjectId(user_id)
        })

        if not prediction:
            return jsonify({'error': 'Prediction not found'}), 404

        prediction['_id'] = str(prediction['_id'])
        prediction['user_id'] = str(prediction['user_id'])
        prediction['created_at'] = prediction['created_at'].isoformat()
        prediction['label'] = 'sarcastic' if prediction.get('is_sarcasm') else 'not sarcastic'
        prediction['conclusion'] = _sarcasm_conclusion(bool(prediction.get('is_sarcasm')))
        if 'sarcastic_percentage' not in prediction or 'non_sarcastic_percentage' not in prediction:
            if prediction.get('is_sarcasm') is True:
                prediction['sarcastic_percentage'] = 100.0
                prediction['non_sarcastic_percentage'] = 0.0
            elif prediction.get('is_sarcasm') is False:
                prediction['sarcastic_percentage'] = 0.0
                prediction['non_sarcastic_percentage'] = 100.0
            else:
                prediction['sarcastic_percentage'] = None
                prediction['non_sarcastic_percentage'] = None
        prediction.pop('confidence', None)
        prediction.pop('raw', None)
        prediction.pop('sentiment_scores', None)

        return jsonify(prediction), 200

    except Exception:
        return _internal_server_error('Prediction lookup failed')


@app.route('/api/feedback', methods=['POST'])
@jwt_required()
def submit_feedback():
    try:
        user_id = get_jwt_identity()
        data = request.get_json(silent=True) or {}
        from bson import ObjectId

        if not data.get('prediction_id') or data.get('is_correct') is None:
            return jsonify({'error': 'prediction_id and is_correct required'}), 400

        prediction = predictions_collection.find_one({
            '_id': ObjectId(data['prediction_id']),
            'user_id': ObjectId(user_id)
        })

        if not prediction:
            return jsonify({'error': 'Prediction not found'}), 404

        feedback = {
            'user_id': ObjectId(user_id),
            'prediction_id': ObjectId(data['prediction_id']),
            'is_correct': data['is_correct'],
            'comment': data.get('comment', ''),
            'created_at': datetime.utcnow()
        }

        result = feedback_collection.insert_one(feedback)

        return jsonify({
            'feedback_id': str(result.inserted_id),
            'message': 'Feedback submitted successfully'
        }), 201

    except Exception:
        return _internal_server_error('Feedback submission failed')


@app.route('/')
def home():
    return _react_page_shell('chat_ui.jsx', 'Sarcasm Detection Chat')


@app.route('/login')
def login_page():
    return _react_page_shell('login.jsx', 'Login - Sarcasm Detector')


@app.route('/signup')
def signup_page():
    return _react_page_shell('signup.jsx', 'Sign Up - Sarcasm Detector')




def _react_page_shell(page_script, title):
    css_file = page_script.replace('.jsx', '.css')
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <link rel="stylesheet" href="/static/css/{css_file}" />
  <script crossorigin src="https://unpkg.com/react@18/umd/react.development.js"></script>
  <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
  <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
</head>
<body>
  <div id="root"></div>
  <script type="text/babel" src="/assets/jsx/{page_script}"></script>
</body>
</html>
"""
    return Response(html, mimetype='text/html')


@app.route('/assets/jsx/<path:filename>')
def jsx_asset(filename):
    allowed_files = {'chat_ui.jsx', 'login.jsx', 'signup.jsx'}
    if filename not in allowed_files:
        return jsonify({'error': 'Asset not found'}), 404

    file_path = os.path.join(app.root_path, 'templates', filename)
    if not os.path.isfile(file_path):
        return jsonify({'error': 'Asset not found'}), 404

    with open(file_path, 'r', encoding='utf-8') as file_obj:
        content = file_obj.read()

    return Response(content, mimetype='text/babel')

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(debug=DEBUG_MODE, host='0.0.0.0', port=5000)



