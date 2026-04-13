# msg_pool.py
import numpy as np
import random

# 仅需数据存储分发功能
class MsgPool():
    def __init__ (self, channel_num : int = 500):
        self.channel = {}
        # 作为服务器收发仿真全局信息
        self.channel["0"] = {} # 保留全局信息频道
        self.agents_info = {}

        self.channel_num = channel_num
        self.channel_id = random.sample(range(1,channel_num *5), channel_num)
        for id in self.channel_id:
            self.channel[f"{id}"] = {}
    def check(self):
        if self.channel_id:
            print(f"MsgPool of {self.channel_num} channels is ready")
    def upload(self, channelid : int, msg : dict):
        if channelid in self.channel_id:
            self.channel[f"{channelid}"] = msg
        elif channelid == 0: # 解析全局信息
            id = msg["id"]
            self.agents_info[f"{id}"] = msg
    def download(self , channelid : int):
        if channelid in self.channel_id:
            return self.channel[f"{channelid}"]
        elif channelid == 0:
            return self.agents_info
        else:
            return None

def main():
    # 测试信息收发
    pool1 = MsgPool(5)
    a1 = {"id" : 1 ,"b" : [1,2]}
    a2 = {"id" : 2 ,"b" : [1,3]}
    a3 = {"id" : 3 ,"b" : [1,4]}
    a4 = {"id" : 4 ,"b" : [1,5]}
    pool1.upload(0,a1)
    pool1.upload(0,a2)
    pool1.upload(0,a3)
    pool1.upload(0,a4)
    b = pool1.download(0)
    c = []
    position = np.array([1,1])
    m_obs = np.array([d["b"] for d in b.values()])
    for obs in m_obs:
        print(obs)
        if np.linalg.norm(obs - position) < 3:
            xy_tuple = (obs[0],obs[1])
            c.append(xy_tuple)
            print(xy_tuple)
    print(c)
    
if __name__ == "__main__":
    main()
