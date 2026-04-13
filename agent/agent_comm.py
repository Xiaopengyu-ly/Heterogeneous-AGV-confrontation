from comm.msg_pool import MsgPool

'''
    [重构版] 通信组件
    不再作为 Agent 的父类，而是作为 Agent 的成员对象 (self.comm_sys)
'''
class CommSystem:
    def __init__(self, agent):
        """
        初始化通信组件
        :param agent: 宿主 Agent 的实例引用，用于访问 id, position 等数据
        """
        self.agent = agent

    def broadcast_msg(self, pool: MsgPool):
        # 通过 self.agent 访问宿主数据
        msg = {
            "id": self.agent.id,
            "pos": self.agent.position,
            "velo": self.agent.velo,
            "r_pos": self.agent.r_point,
            "chanl": self.agent.channel_id
        }
        pool.upload(self.agent.channel_id, msg)

    def recieve_msg(self, pool: MsgPool):
        # 访问宿主的邻居列表
        for nid in self.agent.neighbors_id:
            channel_id = self.agent.neighbors_info["channelid"][f"{nid}"]
            msg = pool.download(channel_id)
            if msg:
                self.agent.neighbors_info["position"][f"{nid}"] = msg["pos"]
                self.agent.neighbors_info["velo"][f"{nid}"] = msg["velo"]
                if not msg["chanl"] == channel_id:
                    self.agent.neighbors_info["channelid"][f"{nid}"] = msg["chanl"]
        
        # 处理目标信息 (Chase Target)
        target_id = self.agent.targets_id
        channel_id = self.agent.targets_info["channelid"][f"{target_id}"] if not target_id == 0 else "Empty"
        msg = pool.download(channel_id)
        if msg:
            self.agent.targets_info["position"][f"{target_id}"] = msg["pos"]
            self.agent.targets_info["velo"][f"{target_id}"] = msg["velo"]
            if not msg["chanl"] == channel_id:
                self.agent.targets_info["channelid"][f"{target_id}"] = msg["chanl"]
        
        # 处理火力目标信息 (Cannon Target)
        cannon_target_id = self.agent.cannon_targets_id
        channel_id = self.agent.cannon_targets_info["channelid"][f"{cannon_target_id}"] if not cannon_target_id == 0 else "Empty"
        msg = pool.download(channel_id)
        if msg:
            self.agent.cannon_targets_info["position"][f"{cannon_target_id}"] = msg["pos"]
            self.agent.cannon_targets_info["velo"][f"{cannon_target_id}"] = msg["velo"]
            if not msg["chanl"] == channel_id:
                self.agent.cannon_targets_info["channelid"][f"{cannon_target_id}"] = msg["chanl"]

    def upload_toPanel(self, pool: MsgPool):
        msg = {
            "id": self.agent.id,
            "pos": self.agent.position,
            "velo": self.agent.velo,
            "angle" : self.agent.theta,
            "WPangle" : self.agent.cannon_theta,
        }
        pool.upload(0, msg)