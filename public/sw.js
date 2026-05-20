const express = require("express");
const cors = require("cors");
const path = require("path");

const app = express();

app.use(cors());
app.use(express.json({ limit: "3mb" }));
app.use(express.static(path.join(__dirname, "public")));

const PORT = process.env.PORT || 3000;

const DEFAULT_SYSTEM = `
Você é o R³ Omnia Core, um orquestrador central de inteligência artificial criado para apoiar Roger na gestão estratégica da Áurea Baby Store e demais projetos.
Sua função é analisar respostas de múltiplas IAs, identificar convergências, divergências, riscos, oportunidades e entregar uma resposta final clara, prática e executável.
Responda em português do Brasil, com postura executiva, direta, estratégica e sem enrolação.
`;

function extractPrompt(body) {
  if (!body) return "";

  if (typeof body === "string") return body;

  if (body.message) return String(body.message);
  if (body.prompt) return String(body.prompt);
  if (body.text) return String(body.text);

  if (Array.isArray(body.messages)) {
    return body.messages
      .map((m) => {
        if (typeof m === "string") return m;
        return `${m.role || "user"}: ${m.content || m.text || ""}`;
      })
      .join("\n");
  }

  return JSON.stringify(body);
}

function extractAgent(body) {
  if (!body) return "Omnia Core";
  return body.agent || body.agentName || body.role || "Omnia Core";
}

function cleanText(text) {
  return String(text || "").trim();
}

function withTimeout(ms) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ms);
  return { controller, timer };
}

async function callOpenAI(prompt, agent) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) throw new Error("OPENAI_API_KEY não configurada");

  const model = process.env.OPENAI_MODEL || "gpt-4.1-mini";

  const { controller, timer } = withTimeout(45000);

  try {
    const response = await fetch("https://api.openai.com/v1/responses", {
      method: "POST",
      signal: controller.signal,
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model,
        input: [
          {
            role: "system",
            content: `${DEFAULT_SYSTEM}\nAgente solicitado: ${agent}`
          },
          {
            role: "user",
            content: prompt
          }
        ]
      })
    });

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      throw new Error(`OpenAI ${response.status}: ${JSON.stringify(data)}`);
    }

    let text = data.output_text;

    if (!text && Array.isArray(data.output)) {
      text = data.output
        .flatMap((item) => item.content || [])
        .map((content) => content.text || "")
        .join("\n");
    }

    text = cleanText(text);

    if (!text) throw new Error("OpenAI respondeu sem texto");

    return {
      provider: "OpenAI",
      model,
      success: true,
      text
    };
  } finally {
    clearTimeout(timer);
  }
}

async function callGemini(prompt, agent) {
  const apiKey = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY;
  if (!apiKey) throw new Error("GEMINI_API_KEY não configurada");

  const model = process.env.GEMINI_MODEL || "gemini-2.5-flash";

  const { controller, timer } = withTimeout(45000);

  try {
    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`,
      {
        method: "POST",
        signal: controller.signal,
        headers: {
          "x-goog-api-key": apiKey,
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          systemInstruction: {
            parts: [
              {
                text: `${DEFAULT_SYSTEM}\nAgente solicitado: ${agent}`
              }
            ]
          },
          contents: [
            {
              role: "user",
              parts: [{ text: prompt }]
            }
          ]
        })
      }
    );

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      throw new Error(`Gemini ${response.status}: ${JSON.stringify(data)}`);
    }

    const text = cleanText(
      data?.candidates?.[0]?.content?.parts
        ?.map((p) => p.text || "")
        .join("\n")
    );

    if (!text) throw new Error("Gemini respondeu sem texto");

    return {
      provider: "Gemini",
      model,
      success: true,
      text
    };
  } finally {
    clearTimeout(timer);
  }
}

async function callAnthropic(prompt, agent) {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) throw new Error("ANTHROPIC_API_KEY não configurada");

  const model = process.env.ANTHROPIC_MODEL || "claude-3-5-haiku-20241022";

  const { controller, timer } = withTimeout(45000);

  try {
    const response = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      signal: controller.signal,
      headers: {
        "x-api-key": apiKey,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model,
        max_tokens: 1200,
        system: `${DEFAULT_SYSTEM}\nAgente solicitado: ${agent}`,
        messages: [
          {
            role: "user",
            content: prompt
          }
        ]
      })
    });

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      throw new Error(`Anthropic ${response.status}: ${JSON.stringify(data)}`);
    }

    const text = cleanText(
      data?.content?.map((c) => c.text || "").join("\n")
    );

    if (!text) throw new Error("Anthropic respondeu sem texto");

    return {
      provider: "Anthropic",
      model,
      success: true,
      text
    };
  } finally {
    clearTimeout(timer);
  }
}

async function callProvider(provider, prompt, agent) {
  try {
    if (provider === "openai") return await callOpenAI(prompt, agent);
    if (provider === "gemini") return await callGemini(prompt, agent);
    if (provider === "anthropic") return await callAnthropic(prompt, agent);

    throw new Error(`Provider desconhecido: ${provider}`);
  } catch (error) {
    console.error(`[${provider}] falhou:`, error.message);

    return {
      provider,
      model: null,
      success: false,
      error: error.message,
      text: ""
    };
  }
}

function availableProviders() {
  const list = [];

  if (process.env.OPENAI_API_KEY) list.push("openai");
  if (process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY) list.push("gemini");
  if (process.env.ANTHROPIC_API_KEY) list.push("anthropic");

  return list;
}

function chooseSynthesizer(results) {
  const priority = (process.env.OMNIA_SYNTHESIZER_PRIORITY || "openai,gemini,anthropic")
    .split(",")
    .map((p) => p.trim().toLowerCase());

  const successfulProviders = results
    .filter((r) => r.success)
    .map((r) => String(r.provider).toLowerCase());

  for (const provider of priority) {
    if (successfulProviders.includes(provider)) return provider;
  }

  return null;
}

function buildSynthesisPrompt(originalPrompt, agent, results) {
  const successful = results.filter((r) => r.success);
  const failed = results.filter((r) => !r.success);

  return `
Você é o R³ Omnia Core em modo ORQUESTRADOR MULTI-IA.

Pergunta original do usuário:
${originalPrompt}

Agente solicitado:
${agent}

Respostas recebidas das IAs:

${successful
  .map(
    (r, index) => `
IA ${index + 1}: ${r.provider}
Modelo: ${r.model}

Resposta:
${r.text}
`
  )
  .join("\n\n")}

Falhas técnicas, se houver:
${failed.map((r) => `${r.provider}: ${r.error}`).join("\n") || "Nenhuma falha relevante."}

Sua tarefa:
1. Compare as respostas.
2. Identifique o que há de melhor em cada uma.
3. Remova contradições, exageros e partes fracas.
4. Gere uma resposta final única, superior às respostas individuais.
5. Seja prático, executivo e direto.
6. Quando for projeto técnico, entregue próximos passos acionáveis.
7. Não diga que é apenas uma síntese; responda como Omnia Core.

Responda agora em português do Brasil.
`;
}

async function synthesizeWithProvider(provider, synthesisPrompt, agent) {
  if (provider === "openai") return await callOpenAI(synthesisPrompt, agent);
  if (provider === "gemini") return await callGemini(synthesisPrompt, agent);
  if (provider === "anthropic") return await callAnthropic(synthesisPrompt, agent);

  throw new Error("Nenhum sintetizador disponível");
}

async function omniaOrchestrate(prompt, agent) {
  const providers = availableProviders();

  if (providers.length === 0) {
    throw new Error("Nenhuma chave de IA configurada. Configure OPENAI_API_KEY, GEMINI_API_KEY ou ANTHROPIC_API_KEY.");
  }

  console.log(`Omnia Core acionando provedores: ${providers.join(", ")}`);

  const results = await Promise.all(
    providers.map((provider) => callProvider(provider, prompt, agent))
  );

  const successful = results.filter((r) => r.success);

  if (successful.length === 0) {
    throw new Error(
      `Todas as IAs falharam: ${results
        .map((r) => `${r.provider}: ${r.error}`)
        .join(" | ")}`
    );
  }

  if (successful.length === 1) {
    return {
      finalText: successful[0].text,
      mode: "single-provider",
      synthesizer: successful[0].provider,
      results
    };
  }

  const synthesisPrompt = buildSynthesisPrompt(prompt, agent, results);
  const synthesizer = chooseSynthesizer(results);

  try {
    const synthesis = await synthesizeWithProvider(
      synthesizer,
      synthesisPrompt,
      "Omnia Core Sintetizador"
    );

    return {
      finalText: synthesis.text,
      mode: "multi-ai-orchestrated",
      synthesizer: synthesis.provider,
      results
    };
  } catch (error) {
    console.error("Falha na síntese final:", error.message);

    const fallbackText = `
Análise consolidada do Omnia Core:

${successful
  .map(
    (r) => `
Fonte: ${r.provider}
${r.text}
`
  )
  .join("\n\n")}

Observação técnica: as IAs responderam, mas a etapa final de síntese automática falhou.`
      .trim();

    return {
      finalText: fallbackText,
      mode: "multi-ai-without-final-synthesis",
      synthesizer: null,
      results
    };
  }
}

app.get("/health", (req, res) => {
  res.json({
    status: "ok",
    app: "R³ Omnia",
    mode: "multi-ai-orchestrator",
    providers: {
      openai: Boolean(process.env.OPENAI_API_KEY),
      gemini: Boolean(process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY),
      anthropic: Boolean(process.env.ANTHROPIC_API_KEY)
    }
  });
});

app.get("/api/providers", (req, res) => {
  res.json({
    ok: true,
    mode: "multi-ai-orchestrator",
    providers: {
      openai: {
        enabled: Boolean(process.env.OPENAI_API_KEY),
        model: process.env.OPENAI_MODEL || "gpt-4.1-mini"
      },
      gemini: {
        enabled: Boolean(process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY),
        model: process.env.GEMINI_MODEL || "gemini-2.5-flash"
      },
      anthropic: {
        enabled: Boolean(process.env.ANTHROPIC_API_KEY),
        model: process.env.ANTHROPIC_MODEL || "claude-3-5-haiku-20241022"
      }
    }
  });
});

async function chatHandler(req, res) {
  try {
    const prompt = extractPrompt(req.body);
    const agent = extractAgent(req.body);

    if (!prompt) {
      return res.status(400).json({
        ok: false,
        error: "Mensagem vazia"
      });
    }

    const orchestration = await omniaOrchestrate(prompt, agent);

    res.json({
      ok: true,
      mode: orchestration.mode,
      synthesizer: orchestration.synthesizer,
      reply: orchestration.finalText,
      response: orchestration.finalText,
      content: orchestration.finalText,
      message: orchestration.finalText,
      debug: {
        providersUsed: orchestration.results.map((r) => ({
          provider: r.provider,
          model: r.model,
          success: r.success,
          error: r.error || null
        }))
      }
    });
  } catch (error) {
    console.error("Erro geral no chat:", error.message);

    res.status(500).json({
      ok: false,
      error: "Erro ao obter resposta",
      details: error.message
    });
  }
}

app.post("/api/chat", chatHandler);
app.post("/chat", chatHandler);
app.post("/api/omnia", chatHandler);

app.get("*", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

app.listen(PORT, () => {
  console.log(`R³ Omnia Core Multi-IA rodando na porta ${PORT}`);
});
