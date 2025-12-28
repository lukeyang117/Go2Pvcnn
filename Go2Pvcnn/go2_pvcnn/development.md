# develoment jornal
1.加入tensorboard，可以记录一下实验结果（解决了，不用管了）
2.把command部分是否要修改一下，改成没和世界坐标关系有关，只有x，y，z，相对于当前机器人坐标系(解决了)
3.把奖励和不同碰撞区分开，碰到不同的物品，惩罚不一样，可能还涉及什么部位碰，我现在是想解决的是区分和不同usd物品碰撞的惩罚，除了足端，其他部位碰到地面都要惩罚，其他所有部位碰到不同usd物品都要惩罚，碰到不同usd惩罚不一样
4.pvcnn的输入输出的问题（输入是不是有空的，输出为什么会有naN or Inf ？[ERROR] PVCNN output contains NaN or Inf! Replacing with zeros.
5.把command和地形的方向确定一下，使用heading_command
