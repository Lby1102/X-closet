# X-Closet 多模态推荐模块

这是唐睿彬个人负责部分的可运行实现，覆盖推荐编排、近期对话压缩、候选过滤、搭配生成、复核修订和推荐查询。项目使用 FastAPI、Pydantic 2 和 asyncio；本地默认使用演示适配器，接口契约可在 `/docs` 和 `/openapi.json` 查看。

## 启动

需要 Python 3.10 或更新版本，建议安装 Python 3.12。安装包可从 [Python 官方 Windows 下载页](https://www.python.org/downloads/windows/)获取。安装后关闭并重新打开 PowerShell，确认 `py --version` 能显示版本号。

先在 PowerShell 切换到本 README 所在的项目目录。首次运行时复制本地环境配置：

```powershell
Copy-Item .env.example .env
notepad .env
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload
```

若 PowerShell 阻止激活虚拟环境，可跳过激活，改用：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```


服务默认只接受预置演示用户 `demo-user`。发送请求时带 `X-User-Id: demo-user`。启动时会读取本文件夹的 `.env`；修改其中的 `ALLOWED_USER_IDS` 可设置本地演示用户列表。生产环境必须替换为项目统一的用户认证和隔离实现。

### 与共享服务接口对接

推荐流程保留了共享服务的接口适配层，不包含组员 Mock 的实现或演示数据。对接关系如下：

| 推荐模块端口 | 共享服务接口 |
| --- | --- |
| `WardrobePort.list_confirmed(user_id)` | `wardrobe.list(user_id)` |
| `WardrobePort.search(user_id, image_id, limit)` | `closet.search(user_id, image_id, limit)` |
| `ProfilePort.get(user_id)` | `memory.get(user_id)` |
| `TrendPort.search_with_warnings(...)` | `trends.search_with_warnings(..., user_id=user_id)` |

`app/integrations.py` 提供组合入口；应用接入时把共享服务实例传入即可。用户 ID 会由推荐流程透传给衣橱、图片检索、画像和趋势接口。当前独立运行仍使用项目自带的 `Demo*` 服务。

默认演示账号仍为 `demo-user`。衣橱、图片、任务和推荐都保存在进程内存中，重启服务会重新生成演示数据。

## 一次推荐请求

```http
POST /api/v1/recommendations
X-User-Id: demo-user
Idempotency-Key: demo-rec-001
Content-Type: application/json
```

```json
{
  "city": "上海",
  "date": "2026-11-25",
  "scene": "commute",
  "request": "简约一点，舒服，别推荐红色",
  "reference_image_id": "img_reference_01",
  "messages": [
    {"role": "user", "text": "我平时喜欢简约通勤风。"},
    {"role": "assistant", "text": "收到。"},
    {"role": "user", "text": "今天通勤，别推荐红色。"}
  ]
}
```

创建请求返回 HTTP 202 和 Job。客户端继续每两秒查询 `GET /api/v1/jobs/{job_id}`，直到 `succeeded` 或 `failed`。成功 Job 的 `result` 包含 `recommendation_id`、搭配和警告；天气快照单独保存供内部记录使用。然后可用 `GET /api/v1/recommendations/{recommendation_id}` 读取已保存结果。重复提交相同用户、路径、幂等键和请求体会复用 Job；同键不同请求体返回 409。

成功结果结构示意：

```json
{
  "data": {
    "job_id": "job_<id>",
    "type": "recommendation",
    "status": "succeeded",
    "stage": "completed",
    "result": {
      "recommendation_id": "rec_<id>",
      "outfits": [
        {"outfit_id": "outfit_<id>", "garment_ids": ["g_top_white", "g_bottom_black"], "reason": "简约通勤组合，适合当前天气。"}
      ],
      "warnings": []
    },
    "error": null,
    "warnings": []
  },
  "request_id": "req_<id>"
}
```

失败仍通过 Job 查询：例如天气依赖失败时 `status=failed`、`result=null`、`error.code=WEATHER_UNAVAILABLE`；衣橱为空为 `WARDROBE_EMPTY`；没有合格搭配或复核两次未通过为 `NO_VALID_OUTFIT`。趋势不可用时任务继续，并在 `warnings` 中写入 `TREND_UNAVAILABLE`。参考图检索不可用也会带警告并回退到合规衣橱候选。

## 代码分层

- `app/schemas.py`：共享请求、响应、Job、衣物、天气、画像、草案和复核 Pydantic 类型，限制场景、消息角色和必填字段。
- `app/context.py`：`compress(messages)` 将最近文本整理成 `scene`、`preferences`、`constraints`、`summary`。禁忌语句单独保留；图像、天气数值和单品 ID 在独立字段里传递，不被摘要替代。
- `app/adapters.py`：天气、衣橱、画像、趋势、生成和复核的接口边界，以及本地演示适配器。复核接口按 v1 使用 `review.check(user_id, draft, weather, profile)` 四个参数；本地演示复核器会按用户 ID 检查草案中的衣物是否在已确认衣橱中。趋势检索带 `user_id`，并通过 `search_with_warnings` 统一返回案例和警告。组员服务就绪后替换这些端口。
- `app/services.py`：推荐主流程。并行读取天气、已确认衣橱、画像、趋势和压缩上下文；天气与其他必要依赖失败时停止，趋势失败时警告后继续。接着按禁忌和简单天气规则过滤，参考图检索结果与过滤结果取交集，交集较少时从合规衣物补齐。生成草案后调用复核，最多修订一次，通过后保存推荐。
- `app/main.py`：HTTP API、用户头校验、请求 ID、幂等键、异步 Job 状态流转，以及推荐和任务查询。

## 适配和当前边界

当前目录没有组内后端代码、真实数据库、Redis/Celery、天气服务、衣橱服务、趋势/偏好服务、徐宇飞的复核器或视觉语言模型凭证。因此默认的 `Demo*` 适配器和 `RuleBasedStylist` 用于本地展示接口与控制流；它不是视觉模型，也不代表真实天气或个人衣橱数据。适配器接口已经留好，接入时应替换演示对象。

Job 和 Recommendation 暂存于进程内存，服务重启后会清空；实际项目应改接郭子骏提供的 PostgreSQL 事务、Redis/Celery Worker 和共享 Job 查询。天气过滤阈值（低于 8℃、高于或等于 28℃）是可调整的演示策略，因为作业材料没有给出精确公式。图片地址在上下文中以 ID/URL 字段传递；接入多模态模型时需由图像存储适配器解析参考图并向模型传递候选图与最近三张灵感图。

本实现对外路径遵循 `/api/v1` 契约。缺少/未知用户为 401，幂等冲突为 409，请求字段无效为 422。天气、衣橱和搭配等后台执行错误写入 Job.error；查询 Job 本身仍返回 200。API Key 不放入代码，应通过本地 `.env` 或部署密钥配置注入。
