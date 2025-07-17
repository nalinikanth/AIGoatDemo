import datetime
import logging
import requests
import argparse
import os
import random
from functools import wraps
import jwt
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, current_app
import json
from models import Product, Category, User, db, Comment, product_categories
import boto3
from botocore.exceptions import NoCredentialsError
from vulnerable_image_processor import process_image
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from sqlalchemy import func, cast, and_, or_, case
from sqlalchemy.dialects.postgresql import JSONB

from flask_cors import CORS
import openai
from threading import Lock
import io
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn


log_handler = logging.StreamHandler()
log_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_handler.setFormatter(formatter)

class FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


parser = argparse.ArgumentParser(description='Flask Application')
parser.add_argument('--db_user', type=str, default='pos_user', help='Database username')
parser.add_argument('--db_password', type=str, default='password123', help='Database password')
parser.add_argument('--db_host', type=str, default='localhost', help='Database host')
parser.add_argument('--db_name', type=str, default='rds-database', help='Database name')
parser.add_argument('--comments_api_gateway', type=str, help='Comments api gateway URL')
parser.add_argument('--similar_images_api_gateway', type=str, help='Similar images api gateway URL')
parser.add_argument('--similar_images_bucket', type=str, help='Similar images bucket name')
parser.add_argument('--get_recs_api_gateway', type=str, help='Get user recommendations api gateway URL')
parser.add_argument('--data_poisoning_bucket', type=str, help='Data Poisoning bucket name')

args = parser.parse_args()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'a_secret_key_that_you_should_change'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

CORS(app, resources={r"/*": {"origins": "*"}})
logging.basicConfig(level=logging.DEBUG)
app.logger.addHandler(FlushHandler())

app.config['SQLALCHEMY_DATABASE_URI'] = f'postgresql://{args.db_user}:{args.db_password}@{args.db_host}:5432/{args.db_name}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

s3 = boto3.client('s3')
sagemaker_client = boto3.client('sagemaker', region_name='us-east-1')

# Dummy user data
users = {
    'babyshark': 'doodoo123'
}


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization').split(" ")[1] if 'Authorization' in request.headers else None
        if not token:
            return jsonify({'message': 'Token is missing!'}), 403

        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = data['username']
        except:
            return jsonify({'message': 'Token is invalid!'}), 403

        return f(current_user, *args, **kwargs)

    return decorated



user_carts = {
    'babyshark': []
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}


def get_product_by_id(product_id):
    return Product.query.get(product_id)

def get_products_by_ids(ids, products):
    id_list = [int(id_str) for id_str in ids.split(',')]
    result = [product for product in products if product.id in id_list]
    return result


@app.route('/')
def home():
    return "Welcome to the AI Goat Store!"


@app.route('/api/recommendations', methods=['GET'])
@token_required
def get_recommendations(current_user):
    user = User.query.filter_by(username=current_user).first_or_404()

    request_payload = {
        "user_id": user.id
    }

    response = requests.post(args.get_recs_api_gateway, json=request_payload, headers={'Content-Type': 'application/json'}, timeout=60)
    response_json = response.json()
    logging.info(f"Response from endpoint: {response_json}")
    if response.status_code == 200:
        body = json.loads(response_json['body'])
        recommended_items_str = body.get('recommended_items', '[]')
        recommended_products_ids = json.loads(recommended_items_str)

        recommended_products_ids = recommended_products_ids[:4]
        user.recommendations = recommended_products_ids

        db.session.commit()
        recommended_products = Product.query.filter(Product.id.in_(recommended_products_ids)).all()
        for product_id in recommended_products_ids:
            product = Product.query.get(product_id)
            if product:
                logging.info(f"Product ID {product_id} exists in database: {product.to_dict()}")
            else:
                logging.info(f"Product ID {product_id} does not exist in database")

        return jsonify([p.to_dict() for p in recommended_products])
    else:
        return jsonify({'error': f'Failed to get recommendations. Response from endpoint: {response_json}'}), 500

@app.route('/api/cart', methods=['GET'])
@token_required
def get_cart(current_user):
    try:
        user = User.query.filter_by(username=current_user.username).first()
        if not user:
            return jsonify({'message': 'User not found'}), 404

        cart_products = Product.query.filter(Product.id.in_(user.cart)).all()
        cart_products_serialized = [product.to_dict() for product in cart_products]

        return jsonify(cart_products_serialized)
    except Exception as e:
        logging.error(f'Error fetching cart: {e}')
        return jsonify({'message': 'Internal Server Error'}), 500


@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.json
        username = data.get('username')
        password = data.get('password')
        if username in users and users[username] == password:
            token = jwt.encode({'username': username, 'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)}, app.config['SECRET_KEY'], algorithm="HS256")
            logging.info(token)
            return jsonify({'message': 'Login successful for {"user_id": 1}', 'token': token})
        return jsonify({'message': 'Invalid credentials'}), 401
    except Exception as e:
        logging.error(f'Error during login: {e}')
        print(f'Error during login: {e}')
        return jsonify({'message': 'Internal Server Error'}), 500


def find_similar_images(query_features, features, top_n=5):
    similarities = {}

    for img_key, img_features in features.items():
        similarity = cosine_similarity([query_features], img_features)[0][0]
        similarities[img_key] = similarity

    sorted_images = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_images[:top_n]


@app.route('/api/analyze-photo', methods=['OPTIONS', 'POST'])
def product_lookup():
    if request.method == 'OPTIONS':
        return '', 204

    matched_products = []
    if request.method == 'POST':
        try:
            app.logger.info('Request Files: %s', request.__dict__)
        except:
            app.logger.error('Error getting request files')

        if 'photo' not in request.files:
            return jsonify({'error': 'No photo uploaded'}), 400

        photo = request.files['photo']
        app.logger.info(f"="*20)
        app.logger.info(photo.filename)

        if not allowed_file(photo.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        else:
            flash('Invalid file format.', 'error')

        bucket_name = args.similar_images_bucket
        api_endpoint = args.similar_images_api_gateway

        if not bucket_name or not api_endpoint:
            return jsonify({'error': 'Missing bucket name or API endpoint'}), 400

        photo_filename = photo.filename
        photo_path = os.path.join('/tmp', photo_filename)
        photo.save(photo_path)

        try:
            s3.upload_file(photo_path, bucket_name, photo_filename)
            app.logger.info(f"Uploaded {photo_filename} to S3 bucket {bucket_name}")
        except NoCredentialsError:
            app.logger.error("Credentials not available")
            return jsonify({'error': 'Credentials not available'}), 500

        with open(photo_path, 'rb') as img_file:
            image_data = img_file.read()
            command_output = process_image(image_data)

        if command_output:
            return jsonify({'metadata_output': command_output}), 400

        features_file_key = 'image_features.pkl'  # S3 key for the features file

        response = s3.get_object(Bucket=bucket_name, Key=features_file_key)
        features = pickle.loads(response['Body'].read())

        payload = {
            'bucket_name': bucket_name,
            'img_key': photo_filename
        }

        response = requests.post(api_endpoint, json=payload)
        if response.status_code != 200:
            return jsonify({'error': 'Error calling the API endpoint'}), response.status_code

        response_json = response.json()
        similar_images_predictions = response_json['predictions'][0]
        similar_images = find_similar_images(similar_images_predictions, features)

        if similar_images:
            similar_image_names = [img[0].split('/')[-1] for img in similar_images]
            if 'orca_doll.jpg' in similar_image_names:
                similar_image_names.remove('orca_doll.jpg')

            cases = [
                case(
                    (Product.images.cast(JSONB).op('@>')(f'[{{"name": "{image_name}"}}]'), index)
                ) for index, image_name in enumerate(similar_image_names)
            ]

            query = Product.query.filter(
                or_(
                    *[
                        Product.images.cast(JSONB).op('@>')(f'[{{"name": "{image_name}"}}]')
                        for image_name in similar_image_names
                    ]
                )
            ).order_by(*cases)
            similar_products = query.all()
            similar_products_serialized = [product.to_dict() for product in similar_products]
            return jsonify(similar_products_serialized)
        else:
            products = Product.query.filter_by(catalog_visibility=True).all()
            if len(products) >= 3:
                random_products = random.sample(products, 3)
            else:
                random_products = products
            random_products_serialized = [product.to_dict() for product in random_products]
            return jsonify(random_products_serialized)

    return jsonify(matched_products)


@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


@app.route('/products/categories', methods=['GET'])
def fetch_categories():
    categories = Category.query.all()
    return jsonify([c.to_dict() for c in categories])

@app.route('/products', methods=['GET'])
def fetch_products():
    category = request.args.get('category')
    include_ids = request.args.get('include')
    if category:
        products = Product.query.filter_by(catalog_visibility=True).join(product_categories).join(Category).filter(Category.slug == category).all()
    else:
        products = Product.query.filter_by(catalog_visibility=True).all()

    if include_ids:
        products = get_products_by_ids(include_ids, products)

    products_serialized = [product.to_dict() for product in products]
    return jsonify(products_serialized)


@app.route('/products/categories/<int:category_id>', methods=['GET'])
def fetch_category_by_id(category_id):
    products = Product.query.filter(Product.categories.contains([{'id': category_id}])).all()
    return jsonify([p.to_dict() for p in products])


@app.route('/products/<int:product_id>/comments', methods=['POST'])
def add_product_comment(product_id):
    data = request.get_json()
    content = data.get('content')
    author = data.get('author')
    is_offensive = data.get('is_offensive')
    probability = data.get('probability')
    if not content:
        return jsonify({"error": "Content is required"}), 400

    product = Product.query.get(product_id)
    if product is None:
        return jsonify({"error": "Id not found"}), 404

    response = requests.post(args.comments_api_gateway, json={"author": str(author), "content": str(content), "is_offensive": is_offensive, "probability": probability})
    if response.status_code != 200:
        return jsonify({"error": "Error calling comment filter service"}), 500

    result = response.json()
    is_offensive_list = result.get("is_offensive", [1])
    if isinstance(is_offensive_list, list) and len(is_offensive_list) > 0 and isinstance(is_offensive_list[0], int) and is_offensive_list[0] in [0, 1]:
        is_offensive_value = is_offensive_list[0]
    else:
        return jsonify({'error': 'Invalid data for "is_offensive". Expected a list with an integer 0 or 1.'}), 400

    if is_offensive_value == 1:
        # Comment is allowed, adding to the db
        comment = Comment(content=content, product_id=product.id)
        db.session.add(comment)
        db.session.commit()

        response_data = comment.to_dict()
        response_data.update({
            "author": author,
            "is_offensive": result.get("is_offensive"),
            "probability": result.get("probability")
        })

        return jsonify(response_data), 201

    else:
        # Comment is not allowed
        response_data = {"error": "Comment is not allowed"}
        response_data.update(result)
        return jsonify(response_data), 400

@app.route('/products/<int:product_id>/comments', methods=['GET'])
def fetch_product_comments(product_id):
    product = Product.query.get(product_id)
    if product is None:
        return jsonify({"error": "Id not found"}), 404

    comments = [comment.to_dict() for comment in product.comments]
    return jsonify(comments)


@app.route('/products/<int:product_id>', methods=['GET'])
def fetch_product_by_id(product_id):
    product = get_product_by_id(product_id)
    if product is None:
        return jsonify({'error': 'Id not found'}), 404
    return jsonify(product.to_dict())  #


@app.route('/api/image-preprocessing', methods=['OPTIONS', 'POST'])
def image_preprocessing():
    return jsonify({'Error': 'Error preprocessing images: An error occurred while trying to preprocess the images. Please try again later.\n For more details, visit our GitHub repository: https://github.com/orcasecurity-research/image-preprocessing-ai-goat'}), 500

@app.route('/api/recommendations-model-endpoint-status', methods=['OPTIONS', 'POST'])
def recommendations_endpoint_status():
    try:
        response = sagemaker_client.describe_endpoint(EndpointName="reccomendation-system-endpoint")
        last_modified_time = response['LastModifiedTime']
        last_modified_time_str = last_modified_time.strftime("%Y-%m-%d %H:%M:%S")
        status = response['EndpointStatus']
        return jsonify({"status": status, "last_modified_time_UTC": last_modified_time_str}), 200
    except Exception as e:
        return jsonify({"error": e}), 500

@app.route('/hints/challenge1/1', methods=['GET'])
def hint_1_1():
    return "Look for some errors"
@app.route('/hints/challenge1/2', methods=['GET'])
def hint_1_2():
    return "Have you heard about malicious PNGs?"

@app.route('/hints/challenge1/3', methods=['GET'])
def hint_1_3():
    return "SSRF - Super Stuffed Rainbow Fluff"
@app.route('/hints/challenge2/1', methods=['GET'])
def hint_2_1():
    return f'{args.data_poisoning_bucket}'

@app.route('/hints/challenge2/2', methods=['GET'])
def hint_2_2():
    return "1,3,4, Let's explore the fluffy store!"
@app.route('/hints/challenge2/3', methods=['GET'])
def hint_2_3():
    return '--acl bucket-owner-full-control'
@app.route('/hints/challenge3/1', methods=['GET'])
def hint_3_1():
    return 'What is the API output?'
@app.route('/hints/challenge3/2', methods=['GET'])
def hint_3_2():
    return "If you hear a BURP, it's probably just the teddy bears having a tea party in their SUITE"
@app.route('/hints/challenge3/3', methods=['GET'])
def hint_3_3():
    return 'When an answer is provided, is there a point in asking a question?'

SECRET_DATA = None

@app.route('/api/inject-secret', methods=['POST'])
def inject_secret():
    global SECRET_DATA
    data = request.get_json() or {}
    secret = data.get('secret', 'SECRET-1234-5678-9012')
    SECRET_DATA = secret
    return jsonify({'status': 'secret_injected', 'secret': secret})

DUMMY_SECRETS = [
    'Credit Card: 4111-1111-1111-1111',
    'Credit Card: 5500-0000-0000-0004',
    'Credit Card: 3400-000000-00009',
    'Credit Card: 6011-0000-0000-0004',
    'Password: hunter2',
    'SSN: 123-45-6789'
]

@app.route('/api/ai-assistant', methods=['POST'])
def ai_assistant():
    if os.getenv('ENABLE_AI_ASSISTANT_DEMO', 'false').lower() != 'true':
        return jsonify({'error': 'AI Assistant demo is disabled by configuration.'}), 403
    data = request.get_json()
    user_input = data.get('user_input', '')
    system_prompt = data.get('system_prompt', 'You are a helpful assistant for a toy store.')
    # Always inject dummy secrets as if memorized from training data
    system_prompt += ' ' + ' '.join(f'(Secret: {s})' for s in DUMMY_SECRETS)
    openai.api_key = os.getenv('OPENAI_API_KEY')
    if not openai.api_key:
        return jsonify({'error': 'OpenAI API key not set'}), 500
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            max_tokens=200
        )
        ai_response = response['choices'][0]['message']['content']
        return jsonify({'response': ai_response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

DATA_LOCK = Lock()
USER_DATA_PATH = 'user_data.json'
PRODUCTS_PATH = 'products.json'
MODEL_PATH = 'recommender_model.json'
MODEL_EXTRACTION_LOG = []

# Helper: Load and save JSON safely

def load_json(path):
    with DATA_LOCK:
        with open(path, 'r') as f:
            return json.load(f)

def save_json(path, data):
    with DATA_LOCK:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

# Poison Data Endpoint
@app.route('/api/poison-data', methods=['POST'])
def poison_data():
    """Injects a poisoned user rating: user 'attacker' gives product 'target_product' a max rating."""
    data = request.get_json() or {}
    user = data.get('user', 'attacker')
    product = data.get('product', 'red_panda_doll')
    rating = data.get('rating', 5)
    user_data = load_json(USER_DATA_PATH)
    user_data.append({'user': user, 'product': product, 'rating': rating})
    save_json(USER_DATA_PATH, user_data)
    return jsonify({'status': 'poisoned', 'user': user, 'product': product, 'rating': rating})

# Retrain Model Endpoint
@app.route('/api/retrain-model', methods=['POST'])
def retrain_model():
    """Retrains a simple recommender: for each user, recommend their highest rated product."""
    user_data = load_json(USER_DATA_PATH)
    model = {}
    for entry in user_data:
        user = entry['user']
        product = entry['product']
        rating = entry['rating']
        if user not in model or rating > model[user]['rating']:
            model[user] = {'product': product, 'rating': rating}
    save_json(MODEL_PATH, model)
    return jsonify({'status': 'model_retrained', 'user_count': len(model)})

# Recommend Endpoint
@app.route('/api/recommend', methods=['POST'])
def recommend():
    """Returns the recommended product for a user based on the current model. Logs query and response for model extraction demo."""
    data = request.get_json() or {}
    user = data.get('user', 'attacker')
    try:
        model = load_json(MODEL_PATH)
        if user in model:
            rec = {'user': user, 'recommendation': model[user]['product'], 'rating': model[user]['rating']}
            MODEL_EXTRACTION_LOG.append({'input': {'user': user}, 'output': rec})
            return jsonify(rec)
        else:
            rec = {'user': user, 'recommendation': None, 'message': 'No recommendation for this user.'}
            MODEL_EXTRACTION_LOG.append({'input': {'user': user}, 'output': rec})
            return jsonify(rec)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-extraction-log', methods=['GET'])
def get_model_extraction_log():
    return jsonify(MODEL_EXTRACTION_LOG)

@app.route('/api/model-extraction-log/clear', methods=['POST'])
def clear_model_extraction_log():
    MODEL_EXTRACTION_LOG.clear()
    return jsonify({'status': 'log_cleared'})

# Load model once at startup
resnet18 = models.resnet18(pretrained=True)
resnet18.eval()
IMAGENET_LABELS = None
with open('imagenet_classes.txt', 'r') as f:
    IMAGENET_LABELS = [line.strip() for line in f.readlines()]

def preprocess_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0)

def predict(image_tensor):
    with torch.no_grad():
        outputs = resnet18(image_tensor)
        _, predicted = outputs.max(1)
        label = IMAGENET_LABELS[predicted.item()]
        return label

# FGSM adversarial attack

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

@app.route('/api/classify-image', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    image_file = request.files['image']
    image_bytes = image_file.read()
    image_tensor = preprocess_image(image_bytes)
    label = predict(image_tensor)
    return jsonify({'prediction': label})

@app.route('/api/adversarial-image', methods=['POST'])
def adversarial_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    epsilon = float(request.form.get('epsilon', 0.1))
    image_file = request.files['image']
    image_bytes = image_file.read()
    image_tensor = preprocess_image(image_bytes)
    image_tensor.requires_grad = True
    output = resnet18(image_tensor)
    init_pred = output.max(1, keepdim=True)[1]
    loss = nn.CrossEntropyLoss()(output, init_pred.squeeze())
    resnet18.zero_grad()
    loss.backward()
    data_grad = image_tensor.grad.data
    perturbed_data = fgsm_attack(image_tensor, epsilon, data_grad)
    adv_label = predict(perturbed_data)
    # Convert perturbed image back to bytes
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    img = perturbed_data.squeeze(0)
    img = inv_normalize(img).clamp(0, 1)
    img_pil = transforms.ToPILImage()(img)
    buf = io.BytesIO()
    img_pil.save(buf, format='PNG')
    buf.seek(0)
    return (
        buf.read(),
        200,
        {
            'Content-Type': 'image/png',
            'X-Adversarial-Prediction': adv_label
        }
    )

@app.route('/api/membership-inference', methods=['POST'])
def membership_inference():
    data = request.get_json() or {}
    user = data.get('user')
    product = data.get('product')
    if not user or not product:
        return jsonify({'error': 'user and product required'}), 400
    user_data = load_json(USER_DATA_PATH)
    in_training = any(entry['user'] == user and entry['product'] == product for entry in user_data)
    confidence = 1.0 if in_training else 0.2
    return jsonify({'user': user, 'product': product, 'confidence': confidence, 'in_training': in_training})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
