import tensorflow as tf
from collections import deque
from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, Dense

INITIAL_EPSILON = 1.0  # ε-greedy法のεの初期値
FINAL_EPSILON = 0.1  # ε-greedy法のεの終値
EXPLORATION_STEPS = 1000000  # ε-greedy法のεが減少していくフレーム数

STATE_LENGTH = 4  # 状態を構成するフレーム数
FRAME_WIDTH = 84  # リサイズ後のフレーム幅
FRAME_HEIGHT = 84  # リサイズ後のフレーム高さ

class Agent():
    def __init__(self, num_actions):
        self.num_actions = num_actions  # 行動数
        self.epsilon = INITIAL_EPSILON  # ε-greedy法のεの初期化
        self.epsilon_step = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS  # εの減少率
        self.time_step = 0  # タイムステップ
        self.repeated_action = 0  # フレームスキップ間にリピートする行動を保持するための変数

        # Replay Memoryの初期化
        self.replay_memory = deque()

        # Q Networkの構築
        self.s, self.q_values, q_network = self.build_network()
        q_network_weights = q_network.trainable_weights

        # Target Networkの構築
        self.st, self.target_q_values, target_network = self.build_network()
        target_network_weights = target_network.trainable_weights

        # 定期的にTarget Networkを更新するための処理の構築
        self.update_target_network = [target_network_weights[i].assign(q_network_weights[i]) for i in xrange(len(target_network_weights))]

        # 誤差関数や最適化のための処理の構築
        self.a, self.y, self.loss, self.grad_update = self.build_training_op(q_network_weights)

        # Sessionの構築
        self.sess = tf.InteractiveSession()

        # 変数の初期化（Q Networkの初期化）
        self.sess.run(tf.initialize_all_variables())

        # Target Networkの初期化
        self.sess.run(self.update_target_network)

    def build_network(self):
        model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu',
                                input_shape=(STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT)))
        model.add(Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu'))
        model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.num_actions))

        s = tf.placeholder(tf.float32, [None, STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT])
        q_values = model(s)

        return s, q_values, model

    def build_training_op(self, q_network_weights):
        a = tf.placeholder(tf.int64, [None])  # 行動
        y = tf.placeholder(tf.float32, [None])  # 教師信号

        a_one_hot = tf.one_hot(a, self.num_actions, 1.0, 0.0)  # 行動をone hot vectorに変換する
        q_value = tf.reduce_sum(tf.mul(self.q_values, a_one_hot), reduction_indices=1)  # 行動のQ値の計算

        # エラークリップ
        error = tf.abs(y - q_value)
        quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)  # 誤差関数

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, momentum=MOMENTUM, epsilon=MIN_GRAD)  # 最適化手法を定義
        grad_update = optimizer.minimize(loss, var_list=q_network_weights)  # 誤差最小化

        return a, y, loss, grad_update

    def get_initial_state(self, observation, last_observation):
        processed_observation = np.maximum(observation, last_observation)
        processed_observation = np.uint8(resize(rgb2gray(processed_observation), (FRAME_WIDTH, FRAME_HEIGHT)) * 255)
        state = [processed_observation for _ in xrange(STATE_LENGTH)]
        return np.stack(state, axis=0)