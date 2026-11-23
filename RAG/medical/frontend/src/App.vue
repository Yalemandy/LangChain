<script setup>
import { ref } from 'vue'

// 页面状态
const question = ref('')
const answer = ref('')
const sources = ref([])
const asking = ref(false)      // 正在提问
const rebuilding = ref(false)  // 正在重建索引
const error = ref('')

// 提问：POST /api/ask
async function ask() {
  const q = question.value.trim()
  if (!q || asking.value) return

  asking.value = true
  error.value = ''
  answer.value = ''
  sources.value = []

  try {
    const res = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q }),
    })
    if (!res.ok) {
      const body = await res.json().catch(() => ({}))
      throw new Error(body.detail || `请求失败(${res.status})`)
    }
    const data = await res.json()
    answer.value = data.answer || '（后端没有返回内容）'
    // 后端在无命中时 sources 返回空字符串，统一成数组
    sources.value = Array.isArray(data.sources) ? data.sources : []
  } catch (e) {
    error.value = e.message || '网络错误，请确认后端已启动'
  } finally {
    asking.value = false
  }
}

// 重建索引：POST /api/rebuild-index（比较慢，需要确认）
async function rebuild() {
  if (rebuilding.value) return
  if (!confirm('重建索引会把向量库清空重灌，耗时几分钟，确定继续吗？')) return

  rebuilding.value = true
  error.value = ''
  try {
    const res = await fetch('/api/rebuild-index', { method: 'POST' })
    if (!res.ok) {
      const body = await res.json().catch(() => ({}))
      throw new Error(body.detail || `重建失败(${res.status})`)
    }
    const data = await res.json()
    alert(data.message || '索引重建完成')
  } catch (e) {
    error.value = e.message || '重建请求失败'
  } finally {
    rebuilding.value = false
  }
}
</script>

<template>
  <div class="page">
    <header class="header">
      <h1>🩺 医疗问答助手</h1>
      <p class="sub">RAG 检索增强：先从医疗问答库检索相似问答，再由大模型生成回答</p>
      <button class="rebuild" :disabled="rebuilding" @click="rebuild">
        {{ rebuilding ? '重建索引中…（几分钟）' : '重建向量索引' }}
      </button>
    </header>

    <div class="ask-bar">
      <input
        v-model="question"
        class="ask-input"
        placeholder="输入你的问题，例如：我有点头痛怎么办"
        :disabled="asking"
        @keyup.enter="ask"
      />
      <button class="ask-btn" :disabled="asking || !question.trim()" @click="ask">
        {{ asking ? '思考中…' : '提问' }}
      </button>
    </div>

    <p v-if="error" class="error">⚠️ {{ error }}</p>

    <section v-if="asking" class="loading">正在检索知识库并生成回答…</section>

    <section v-else-if="answer" class="card answer-card">
      <h2>回答</h2>
      <p class="answer">{{ answer }}</p>
    </section>

    <section v-if="sources.length" class="card">
      <h2>参考来源（{{ sources.length }} 条）</h2>
      <div v-for="s in sources" :key="s.rank" class="source">
        <div class="source-head">
          <span class="rank">#{{ s.rank }}</span>
          <span class="score">相关度 {{ (s.score * 100).toFixed(1) }}%</span>
        </div>
        <p class="source-q">问：{{ s.metadata?.question }}</p>
        <p class="source-t">{{ s.text }}</p>
      </div>
    </section>

    <p v-else-if="answer" class="no-source">未检索到相似问答</p>
  </div>
</template>

<style scoped>
.page {
  max-width: 760px;
  margin: 0 auto;
  padding: 32px 20px 60px;
  font-family: -apple-system, "PingFang SC", sans-serif;
  color: #1f2d3d;
}
.header h1 { margin: 0 0 6px; font-size: 26px; }
.sub { margin: 0 0 16px; color: #8492a6; font-size: 13px; }
.rebuild {
  background: #fff; border: 1px solid #d3dce6; border-radius: 8px;
  padding: 6px 14px; font-size: 13px; cursor: pointer; color: #475669;
}
.rebuild:disabled { opacity: .6; cursor: not-allowed; }

.ask-bar { display: flex; gap: 10px; margin: 12px 0 20px; }
.ask-input {
  flex: 1; padding: 12px 16px; font-size: 15px;
  border: 1px solid #d3dce6; border-radius: 10px; outline: none;
}
.ask-input:focus { border-color: #409eff; }
.ask-btn {
  padding: 0 24px; font-size: 15px; color: #fff;
  background: #409eff; border: none; border-radius: 10px; cursor: pointer;
}
.ask-btn:disabled { background: #a0cfff; cursor: not-allowed; }

.error {
  background: #fef0f0; color: #f56c6c; border: 1px solid #fde2e2;
  border-radius: 8px; padding: 10px 14px; font-size: 14px;
}
.loading { color: #8492a6; font-size: 14px; padding: 20px 0; text-align: center; }

.card {
  background: #fff; border: 1px solid #e6ebf2; border-radius: 12px;
  padding: 18px 20px; margin-bottom: 16px;
}
.card h2 { margin: 0 0 10px; font-size: 16px; color: #475669; }
.answer { margin: 0; white-space: pre-wrap; line-height: 1.7; font-size: 15px; }

.source { border-top: 1px dashed #e6ebf2; padding: 12px 0; }
.source:first-of-type { border-top: none; }
.source-head { display: flex; justify-content: space-between; margin-bottom: 6px; }
.rank { font-weight: 600; color: #409eff; }
.score { color: #8492a6; font-size: 13px; }
.source-q { margin: 0 0 4px; font-weight: 600; font-size: 14px; }
.source-t { margin: 0; color: #6b7a90; font-size: 13px; line-height: 1.6; }
.no-source { color: #c0c4cc; font-size: 13px; text-align: center; }
</style>
