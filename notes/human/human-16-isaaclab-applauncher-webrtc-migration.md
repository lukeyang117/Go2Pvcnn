# Human IsaacLab AppLauncher WebRTC Migration Guide

## 导航

- 文档类型：`human` 远程 WebRTC / IsaacLab 迁移手册
- 对应 AI 文档：暂无
- 上一篇：[human-15-raw-kinematic-planner-and-trajectory-training-summary.md](human-15-raw-kinematic-planner-and-trajectory-training-summary.md)
- 相关命令文档：[human-12-batched-planner-train-viewer-commands.md](human-12-batched-planner-train-viewer-commands.md)
- 相关诊断总结：[human-14-batched-planner-viewer-diagnostics-summary.md](human-14-batched-planner-viewer-diagnostics-summary.md)
- 总索引：[../index.md](../index.md)

## 这篇文档解决什么问题

这篇文档只讲一个很具体的问题：

- 你把 `Go2Pvcnn` 或相关 viewer 代码迁移到另一台服务器后，
- 远程启动 Isaac Sim / IsaacLab WebRTC viewer，
- 本地浏览器或客户端能弹出窗口，但没有画面，或者一会儿就断开，
- 同时服务端日志里出现 `nvstPushStreamData error`、`timeout for eye`、`GLFW initialization failed` 之类的输出。

这时除了项目内的 viewer 脚本改动之外，还需要检查并修改 `IsaacLab` 自身的 `AppLauncher`。

## 一句话结论

迁移到新服务器时，如果远程 WebRTC viewer 仍然黑屏或在连接后报：

- `carb.livestream-rtc.plugin nvstPushStreamData error ... 0x800b0000`
- `carb.livestream-rtc.plugin nvstPushStreamData timeout for eye 0`

优先检查 `IsaacLab/source/isaaclab/isaaclab/app/app_launcher.py`：

- `livestream=2` 时不能同时启用
  - `omni.kit.livestream.webrtc`
  - `omni.services.livestream.nvcf`
- 正确做法是只保留后者，并把 WebRTC Kit 参数统一放到 `_resolve_experience_file()` 里追加。

## 症状长什么样

本轮实际遇到的现象是：

1. 服务端命令能正常启动。
2. 本地可视化客户端或浏览器能弹出窗口。
3. 但画面是黑的，或者连上一小会儿后断掉。
4. 服务端日志出现：
   - `nvstPushStreamData timeout for eye 0`
   - `nvstPushStreamData error ... 0x800b0000`
   - `NVST_CCE_DISCONNECTED`
5. 同时还可能看到：
   - `GLFW initialization failed`
   - `add_menu_items ... cannot change delegate`

需要特别注意：

- `GLFW initialization failed` 在 headless WebRTC 场景下不一定是根因。
- 这次真正的关键根因，是 `AppLauncher` 给 `livestream=2` 追加了两套 WebRTC extension。

## 要改哪个文件

目标文件是：

- `IsaacLab/source/isaaclab/isaaclab/app/app_launcher.py`

如果你的 IsaacLab 不在当前仓库内，而是在旁边单独 clone 的目录，实际路径类似：

- `/mnt/mydisk/lhy/IsaacLab/source/isaaclab/isaaclab/app/app_launcher.py`

迁移到新服务器后，只要 IsaacLab 的目录结构类似，都去找这一个文件。

## 根因到底是什么

### 1. 出问题前的逻辑

在未修改的 `AppLauncher` 里，`livestream=2` 的参数是分两次追加的。

第一处在：

- `_resolve_livestream_settings()`

它会先追加：

```text
--/app/livestream/allowResize=false
--enable
omni.kit.livestream.webrtc
```

第二处在：

- `_resolve_experience_file()`

它又会追加：

```text
--/app/livestream/publicEndpointAddress=<PUBLIC_IP>
--/app/livestream/port=49100
--enable
omni.services.livestream.nvcf
```

也就是说，`livestream=2` 实际上启用了两套 livestream extension。

### 2. 为什么这会出问题

源码注释自己就写了：

- `Only one livestream extension can be enabled at a time`

但旧逻辑等于：

1. 先启用 `omni.kit.livestream.webrtc`
2. 后面再启用 `omni.services.livestream.nvcf`

这样在客户端真正连上、开始推流时，就容易出现：

- stream push timeout
- stream push error
- connect/disconnect 反复震荡

这和“根本起不来”不是一类问题，而是“能启动、能连上，但渲染流推送阶段冲突了”。

## 应该怎么改

### 修改原则

把 `livestream` 的职责拆干净：

1. `_resolve_livestream_settings()`
   只负责：
   - 解析 `livestream` 值
   - 做合法性检查
   - 不再往 `sys.argv` 里塞 WebRTC extension

2. `_resolve_experience_file()`
   统一负责：
   - 依据 `PUBLIC_IP` 构造 WebRTC Kit 参数
   - 只追加一套真正要用的 livestream extension

### 需要保留的最终行为

对于 `livestream=2`，最后只保留下面这组参数：

```text
--/app/livestream/publicEndpointAddress=<PUBLIC_IP>
--/app/livestream/port=49100
--enable
omni.services.livestream.nvcf
```

不要再保留：

```text
--enable
omni.kit.livestream.webrtc
```

## 具体改法

### 1. `_resolve_livestream_settings()` 里做什么

这个函数里保留：

- `LIVESTREAM` 环境变量读取
- 输入参数优先级处理
- `self._livestream` 赋值
- `self._livestream_args = []`

这个函数里删除：

- 所有 `if self._livestream >= 1:` 的 extension 追加逻辑
- `sys.argv += self._livestream_args`

改完后，这里应该是一个“无副作用”的解析函数。

### 2. `_resolve_experience_file()` 里做什么

这个函数里保留并继续使用：

- `public_ip_env = os.environ.get("PUBLIC_IP", "127.0.0.1")`

然后在 `self._livestream == 2` 时，只追加：

```text
--/app/livestream/publicEndpointAddress=<PUBLIC_IP>
--/app/livestream/port=49100
--enable
omni.services.livestream.nvcf
```

这里仍然可以保留：

- `sys.argv += self._livestream_args`

因为真正的 WebRTC extension 现在只会在这里追加一次。

## 推荐补丁思路

如果你在新服务器上手工改，最稳的思路就是：

1. 打开 `app_launcher.py`
2. 找到 `_resolve_livestream_settings()`
3. 删掉里面 `livestream>=1` 的 extension 追加逻辑
4. 保留 `self._livestream_args = []`
5. 找到 `_resolve_experience_file()`
6. 确认 `livestream=2` 只追加 `omni.services.livestream.nvcf`
7. 确认没有别的地方再给 `livestream=2` 追加 `omni.kit.livestream.webrtc`

## 和项目内 viewer 脚本的关系

只改 IsaacLab 还不够，这次稳定起来其实依赖两层配合。

### 第一层：项目内 viewer 脚本

当前项目里：

- [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)

已经增加了：

- `--webrtc-public-ip`
- `--webrtc-port`
- `--no-webrtc-auto-public-ip`

并且会在 `--livestream 2` 时优先设置：

- `PUBLIC_IP`

如果 `PUBLIC_IP` 没设，IsaacLab 很容易默认广告：

- `127.0.0.1`

这对远程浏览器是错的。

### 第二层：IsaacLab AppLauncher

即便 `PUBLIC_IP` 正确，如果 `AppLauncher` 还同时启用两套 livestream extension，还是会在连接后推流阶段翻车。

所以完整迁移要求是：

1. 项目内 viewer 脚本能正确设置 `PUBLIC_IP`
2. IsaacLab 只启用一套 WebRTC livestream extension

缺一不可。

## 迁移到新服务器时的最小检查清单

每次换服务器，建议按这个顺序检查：

1. `Go2Pvcnn/extension/viz/go2_foostep_planner.py` 是否包含：
   - `--webrtc-public-ip`
   - `--webrtc-port`
   - `PUBLIC_IP` 预设逻辑
2. `IsaacLab/source/isaaclab/isaaclab/app/app_launcher.py` 是否已经移除
   - `omni.kit.livestream.webrtc`
   这套 `livestream=2` 追加逻辑
3. 启动命令是否显式传了：
   - `--livestream 2`
   - `--webrtc-public-ip <服务器可访问 IP>`
4. 是否没有外层 `timeout` 把 viewer 进程杀掉
5. 客户端连接后，服务端日志里是否出现：
   - `Streaming server started.`
   - `[Viewer][Plan]`
   - `[Viewer][Playback]`
6. 服务端日志里是否没有再出现：
   - `nvstPushStreamData timeout`
   - `nvstPushStreamData error`

## 推荐启动命令（MPC Planner）

当前项目里远程 WebRTC viewer（`mpc` backend）推荐命令是：

```bash
cd /mnt/mydisk/lhy/testPvcnnWithIsaacsim

PYTHONPATH=/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/mnt/mydisk/lhy/testPvcnnWithIsaacsim \
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
Go2Pvcnn/extension/viz/go2_foostep_planner.py \
  --headless \
  --livestream 2 \
  --webrtc-public-ip 172.31.179.75 \
  --device cuda:2 \
  --num_envs 1 \
  --terrain task \
  --planner-backend mpc
```

迁移到新服务器时，最重要的是替换：

- `--webrtc-public-ip`
- Python 环境路径
- `cuda` 设备号

如果你要直接启动训练并验证 `mpc` backend，推荐命令是：

```bash
cd /mnt/mydisk/lhy/testPvcnnWithIsaacsim

PYTHONPATH=/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/mnt/mydisk/lhy/testPvcnnWithIsaacsim \
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
Go2Pvcnn/scripts/train.py \
  --headless \
  --device cuda:2 \
  --num_envs 4096 \
  --max_iterations 1 \
  --experiment teacher_elevation_trajectory \
  --planner-backend mpc
```

## 改完后怎么验证

最少做三层验证。

### 1. 静态检查

- `python -m py_compile app_launcher.py`
- `git diff --check`

这层只能证明语法和 patch 质量没问题。

### 2. 参数层检查

确认 `livestream=2` 最终只会追加一套 WebRTC 参数：

- 有 `publicEndpointAddress`
- 有 `port=49100`
- 有 `omni.services.livestream.nvcf`
- 没有 `omni.kit.livestream.webrtc`

### 3. 真实 smoke

真正跑一次远程 viewer，观察日志是否出现：

- `Streaming server started.`
- `[Viewer][Plan]`
- `[Viewer][Playback]`

同时确认没有：

- `nvstPushStreamData timeout`
- `nvstPushStreamData error`

## 哪些 warning 不要先误判成根因

这次经验里，下列 warning 不是第一优先级根因：

- `GLFW initialization failed`
- `add_menu_items ... cannot change delegate`
- `/rtx-defaults/... No setting was found ...`

它们可能出现，但不等于就是黑屏主因。

真正更值得先排查的是：

1. `PUBLIC_IP` 是否正确
2. `livestream=2` 是否启用了两套 extension
3. 客户端连接后是否在推流阶段报 `nvstPushStreamData`

## 这次迁移经验的边界

这篇文档只覆盖：

- IsaacLab `AppLauncher` 的 WebRTC extension 冲突问题

它不覆盖：

- 浏览器端 WebRTC 调试细节
- 端口转发策略
- 防火墙规则
- 项目内 viewer teleop 输入链
- viewer 持续运行到 `Ctrl-C` 的主循环修改

这些仍然要结合：

- [human-12-batched-planner-train-viewer-commands.md](human-12-batched-planner-train-viewer-commands.md)
- [human-14-batched-planner-viewer-diagnostics-summary.md](human-14-batched-planner-viewer-diagnostics-summary.md)
- 以及对应 `notes/log/` 里的验证记录一起看。

## 相关证据

- [../log/2026-05-06-2054-isaaclab-livestream-dedup-fix.md](../log/2026-05-06-2054-isaaclab-livestream-dedup-fix.md)
- [../log/2026-05-06-2011-viewer-webrtc-public-ip-fix.md](../log/2026-05-06-2011-viewer-webrtc-public-ip-fix.md)
- [../log/2026-05-06-2106-viewer-persistent-loop-fix.md](../log/2026-05-06-2106-viewer-persistent-loop-fix.md)
