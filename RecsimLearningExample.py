import numpy as np
from gym import spaces

from recsim import document
from recsim import user
from recsim.simulator import environment
from recsim.simulator import recsim_gym


# =========================================================
# 1. Document
# =========================================================
#clearly以下是有关Document的内容
class SimpleDocument(document.AbstractDocument):#这里括号里的参数表示这是一个继承自父类的类，
    #这里的document.表示父类所在的地方，在这里我们从recsim中引入了document这个东西，父类的名字就叫做abstract

    #似乎是用来指定一个document有哪些信息的。
    def __init__(self, doc_id, topic, quality):#在python中，_init_大概作用类似于类的构造函数
        #这里有四个参数,self(init的第一个参数)固定是用来指代这个类实例自身的，doc_id应该就是这个document的唯一标示id
        #topic应该是和主题相关的信息，quality是document质量。一个简单的document模型中只包含topic/质量两个信息
        self.topic = float(topic)      # 0.0 or 1.0
        self.quality = float(quality)  # in [0, 1]
        #这里就是用我们传进来的参数给self赋值

        super(SimpleDocument, self).__init__(doc_id)#super是用来调用父类的方法的函数，这里暂时没搞懂是在干什么

    def create_observation(self):
        #对于一个没有额外标注的类方法，其第一个参数一定就是用来指代这个类实例本身的
        return np.array([self.topic, self.quality], dtype=np.float32)#返回一个数组，这个数组的内容：主题，质量
        #dtype=用于指定这个数组中数据类型的类别

    @classmethod#类方法声明，一个类方法的第一个参数指向类本身，直接读取这个类的性质
    def observation_space(cls):#给强化学习框架读取的，必须要有！逻辑上没有什么用
        return spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )


class SimpleDocumentSampler(document.AbstractDocumentSampler):
    #这个东西叫sampler，likely这是取样用的
    def __init__(self, doc_ctor=SimpleDocument, seed=0):
        #这里的=的意思是默认参数，如果没有传参数，就用等于号后面的内容作为参数。
        #在这里，doc_ctor被赋予的默认值是一个类，代表之后我们可以用doc_ctor来替代使用SimpleDocument
        #是的，在python中，一个抽象类是可以作为参数的。
        super(SimpleDocumentSampler, self).__init__(doc_ctor, seed)#在python中，子类的构造函数，需要实现父类的构造函数
        #seed传给父构造函数，支持后面的几个随机方法

    def sample_document(self):
        #这一步，当然就是在进行取样的过程了。
        doc_id = self._rng.randint(0, 10**9)#随机给一个id？
        topic = self._rng.randint(0, 2)
        quality = self._rng.random_sample()#这里出现了三个没有定义的方法，怎么回事呢？
        #它们在父类中被定义。总之就是随机取了
        return self._doc_ctor(doc_id=doc_id, topic=topic, quality=quality)
        #这就是抽象类参数的意义，sampler最终返回一个simple document类


# =========================================================
# 2. User state
# =========================================================
#这里模仿document，同样是分为两类：用户的信息/对用户的sampler
class SimpleUserState(user.AbstractUserState):#继承自recsim中的抽象用户状态
    NUM_FEATURES = 2#用来指示user层状态的维度。这个实例里可以不写，如果用神经网络训练的话，写一个这个可方便读写
    #代表一个局部变量，不是成员

    def __init__(self, preference, fatigue, time_budget):
        #用户的实际三个特点：偏好（用户向量），疲劳度，时间预算
        self.preference = float(preference)   # 0.0 or 1.0
        self.fatigue = float(fatigue)         # in [0, 1]
        self.time_budget = int(time_budget)#初始化赋值咯

    def create_observation(self):
        return np.array([self.preference, self.fatigue], dtype=np.float32)#只能观测到用户的偏好和用户的疲劳度

    @staticmethod#声明这是一个静态方法
    def observation_space():
        return spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )#同样，是给强化学习框架看的，这里类方法和静态方法并没有本质差别


class SimpleUserSampler(user.AbstractUserSampler):
    def __init__(self, user_ctor=SimpleUserState, seed=0):
        super(SimpleUserSampler, self).__init__(user_ctor=user_ctor, seed=seed)

    def sample_user(self):
        preference = self._rng.randint(0, 2)
        return self._user_ctor(
            preference=preference,
            fatigue=0.0,
            time_budget=10
        )#在给用户做sampler的时候，随机给出preference，疲劳度和时间预算在我们的这个时间里是直接给出的


# =========================================================
# 3. Response
# =========================================================
class SimpleResponse(user.AbstractResponse):
    #用户的反应应有的结构
    def __init__(self, clicked=False, watch_time=0.0):
        #用户的反应有两个元素，是否点击，以及点击之后看了多久。（这不适用于短视频，后面我们再修改）
        self.clicked = int(clicked)
        self.watch_time = float(watch_time)

    def create_observation(self):
        return {
            "clicked": self.clicked,
            "watch_time": np.array([self.watch_time], dtype=np.float32),
            #这里返回的是一个字典，字典是python中特有的一种数据结构，类似于其他语言中的hash 表，代表一个键值映射到一个对象上

        }

    @staticmethod
    def response_space():
        return spaces.Dict({
            "clicked": spaces.Discrete(2),
            "watch_time": spaces.Box(
                low=np.array([0.0], dtype=np.float32),
                high=np.array([100.0], dtype=np.float32),
                dtype=np.float32,
            ),
        })#好吧，暂时不清楚这一部分是干什么的，我觉得可以先跳过


# =========================================================
# 4. User model
# =========================================================
class SimpleUserModel(user.AbstractUserModel):
    def __init__(
        #依旧是在初始化，只不过这里换了一种格式（类似于C++中的那样）
        self,
        slate_size,#推荐器一次推荐给用户的视频数。
        user_state_ctor=SimpleUserState,#用户的状态
        response_model_ctor=SimpleResponse,#回应
        seed=0
    ):
        sampler = SimpleUserSampler(user_ctor=user_state_ctor, seed=seed)#随机取一个用户到sampler里
        super(SimpleUserModel, self).__init__(
            response_model_ctor=response_model_ctor,
            user_sampler=sampler,
            slate_size=slate_size
        )

    def simulate_response(self, slate_documents):
        #这个函数用来存放模拟用户行为的具体函数，

        responses = []#建立一个空列表
        for doc in slate_documents:#对每一个在传入document列表的document而言：
            doc_obs = doc.create_observation()#接收这个文档能够观察到的内容：doc_obs
            topic = doc_obs[0]
            quality = doc_obs[1]#提取出这个obs的质量和主题

            match = 1.0 if topic == self._user_state.preference else 0.0#进行一个质量匹配，如果topic和preference是一致的，那么就
            score = 2.0 * match + 0.5 * float(quality) - 1.0 * self._user_state.fatigue#计算最终得分：
            #2*主题匹配+0.5*质量-疲劳

            clicked = score > 0.5#如果计算出最终得分大于0.5，则记录一次点击，否则视作不进行点击。
            watch_time = max(0.0, score * 5.0) if clicked else 0.0#如果有点击，则记录观看时间，否则无观看时间
            #这里其实max不是很有必要，如果score小于0是不会进点击的。

            responses.append(
                #responses列表往下顺延一位：记录是否点击和观看时间。
                self._response_model_ctor(
                    clicked=clicked,
                    watch_time=watch_time
                )
            )
        return responses

    def update_state(self, slate_documents, responses):
        #更新用户的状态，
        clicked_any = any(r.clicked for r in responses)#any:python内置函数，判断是否至少有一个参数为true
        #这里是一次更新一组response，是有点问题的，不过作为一个prototype我们也不深究了。

        if clicked_any:
            self._user_state.fatigue = min(1.0, self._user_state.fatigue + 0.12)#如果点击了任何视频，那么疲劳度+0.12但是不超过1
        else:
            self._user_state.fatigue = max(0.0, self._user_state.fatigue - 0.05)#如果没有点击任何视频，疲劳度-0.05，但是不小于0

        self._user_state.time_budget -= 1#时间预算减一

    def is_terminal(self):
        return self._user_state.time_budget <= 0#判断是否是terminal，判断依据为时间预算是否还有任何剩余


# =========================================================
# 5. Reward
# =========================================================
def total_watch_time_reward(responses):
    return sum(r.watch_time for r in responses)
#先看这个函数本身的逻辑：传入用户的一组response，返回总观看时间，这里只是定义了我们的reward目标，真正的reward在后面实现。


# =========================================================
# 6. Build environment
# =========================================================
#前面的都不算数，这个算是真正的把我们的所有东西打包进一个recsim环境里。
def make_env(num_candidates=5, slate_size=1):
    #这里的两个参数：num_candidates表示agent一次能选择的视频数量。slate_size表示从中选出来推荐给用户的数量
    user_model = SimpleUserModel(
        #创建一个usermodel的实例，赋值给user model
        slate_size=slate_size,#是的，这就是我最讨厌python的地方，太智能了，导致读的时候比较困难
        #这里前面一个slate_size是user model的参数，后面一个是我们这个函数传进来的参数。
        user_state_ctor=SimpleUserState,
        #创建user_state,这里的我们的user state完全是指定实现的，所以不用传入任何参数，等于我们现在有了一个表示user_state的东西

        response_model_ctor=SimpleResponse,
        #创建一个response model，response当然不应该是提前指定的，而应该是在实际反应过程中填入的。
        #老实说这个文件的结构真不咋地

        seed=0
    )

    doc_sampler = SimpleDocumentSampler(
        #实例化一个sampler for document
        doc_ctor=SimpleDocument,
        seed=0
    )

    raw_env = environment.SingleUserEnvironment(
        #SingleUserEnvironment是Recsim里已经实现的一个类。 我们用这个东西来implement我们的整个环境
        user_model=user_model,#环境需要用户模型，sure，我们将前面我们建立的用户模型赋值到这里
    #包含了用户状态和用户反应

        document_sampler=doc_sampler,#需要一个document sampler

        num_candidates=num_candidates,#给定agent一次能够看到的推荐视频的数量
        slate_size=slate_size,#将一次推荐的数量指定

        resample_documents=True#resample_documents是recsim的一个参数，代表每一轮step重新生成document candidate列表。
    )

    env = recsim_gym.RecSimGymEnv(
        #recsim_gym.RecsimGymEnv代表最终的环境
        raw_environment=raw_env,#第一个参数，我们打包前面的原始环境，包含用户状态，用户反应，document sampler，agent一次能看到的视频的数量和推荐的数量
        reward_aggregator=total_watch_time_reward#我们将我们需要最大化的因子传输给reward_aggregator
    )
    return env


# =========================================================
# 7. A trivial agent: always pick candidate 0
# =========================================================
def pick_first_candidate_action(slate_size=1):
    # RecSim environment.step expects a slate (list/array), not a scalar
    return list(range(slate_size))


# =========================================================
# 8. Main
# =========================================================
if __name__ == "__main__":
    #接下来让我们基于主函数看一下整个recsim工作的逻辑。
    slate_size = 1#定义slate_size，这个最终agent给用户看的视频数量的参数
    env = make_env(num_candidates=5, slate_size=slate_size)#在make_env中，我们完成实例化（前面给出的是实例化方法）
    #创建一个env，这回让一切都准备好。make_env这个函数里面完成了什么工作呢？
    #返回raw_env&强化学习目标函数
    #raw_env:user_model  document_sampler  环境参数：能看到的document和最终生成的document数
    #user_model:属于simpleUserModel类别。这个类别包含了一个用户状态模型&一个反应模型

    obs = env.reset()#开启一个新的episode，可以理解为初始化环境
    print("===== RESET =====")
    print(obs)

    for t in range(10):
        action = pick_first_candidate_action(slate_size=slate_size)
        obs, reward, done, info = env.step(action)#.step代表在当前的recsim环境中进行下一轮工作。

        print(f"\n===== STEP {t} =====")
        print("action:", action)
        print("reward:", reward)
        print("obs:", obs)
        print("done:", done)

        if done:
            print("\nEpisode finished.")
            break