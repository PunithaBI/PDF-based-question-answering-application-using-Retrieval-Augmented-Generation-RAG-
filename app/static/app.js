"use strict";

const API = "";

const state = {
    documents: [],
    selectedDocument: null,
    lastAnswer: null
};

const $ = (id) => document.getElementById(id);


// --------------------------------------------------
// API
// --------------------------------------------------

async function api(path, options = {}) {
    const response = await fetch(`${API}${path}`, options);

    const contentType = response.headers.get("content-type") || "";

    let data;

    if (contentType.includes("application/json")) {
        data = await response.json();
    } else {
        data = await response.text();
    }

    if (!response.ok) {
        const message =
            typeof data === "object" && data !== null
                ? data.detail || JSON.stringify(data)
                : data;

        throw new Error(message || `HTTP ${response.status}`);
    }

    return data;
}


function escapeHTML(value) {
    return String(value ?? "").replace(/[&<>"']/g, (character) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;"
    })[character]);
}


function formatNumber(value) {
    return Number(value || 0).toLocaleString();
}


function formatDate(value) {
    if (!value) return "—";

    const date = new Date(value);

    if (Number.isNaN(date.getTime())) {
        return String(value);
    }

    return date.toLocaleString();
}


function showNotice(message, type = "info") {
    const notice = $("notice");

    notice.textContent = message;
    notice.className = `notice ${type}`;

    window.scrollTo({
        top: 0,
        behavior: "smooth"
    });
}


function hideNotice() {
    $("notice").classList.add("hidden");
}


function setLoading(elementId, loading) {
    $(elementId).classList.toggle("hidden", !loading);
}


function setButtonLoading(buttonId, loading, loadingText) {
    const button = $(buttonId);

    if (loading) {
        button.dataset.originalText = button.textContent;
        button.textContent = loadingText;
        button.disabled = true;
    } else {
        button.textContent =
            button.dataset.originalText || button.textContent;

        button.disabled = false;
    }
}


// --------------------------------------------------
// NAVIGATION
// --------------------------------------------------

function showView(viewName) {
    document.querySelectorAll(".view").forEach((view) => {
        view.classList.remove("active");
    });

    document.querySelectorAll(".nav-item").forEach((item) => {
        item.classList.remove("active");
    });

    const view = $(`view-${viewName}`);

    if (!view) return;

    view.classList.add("active");

    const navItem = document.querySelector(
        `.nav-item[data-view="${viewName}"]`
    );

    if (navItem) {
        navItem.classList.add("active");
    }

    const titles = {
        overview: "Overview",
        documents: "Documents",
        ask: "Ask a question",
        history: "Question history",
        evaluation: "Evaluation",
        activity: "System activity"
    };

    $("pageTitle").textContent =
        titles[viewName] || "PDF Evidence Explorer";

    hideNotice();

    if (viewName === "overview") {
        refreshOverview();
    }

    if (viewName === "documents") {
        refreshDocuments();
    }

    if (viewName === "ask") {
        populateDocumentSelect();
    }

    if (viewName === "history") {
        refreshHistory();
    }

    if (viewName === "evaluation") {
        refreshEvaluations();
    }

    if (viewName === "activity") {
        refreshActivity();
    }
}


document.querySelectorAll(".nav-item").forEach((item) => {
    item.addEventListener("click", () => {
        showView(item.dataset.view);
    });
});


document.querySelectorAll("[data-go]").forEach((button) => {
    button.addEventListener("click", () => {
        showView(button.dataset.go);
    });
});


// --------------------------------------------------
// API HEALTH
// --------------------------------------------------

async function checkHealth() {
    const status = $("connectionStatus");

    try {
        await api("/api/health");

        status.textContent = "API connected";

        status.previousElementSibling.style.background = "#62d49b";

    } catch (error) {
        status.textContent = "API unavailable";

        status.previousElementSibling.style.background = "#f25b24";
    }
}


// --------------------------------------------------
// OVERVIEW
// --------------------------------------------------

async function refreshOverview() {
    try {
        const [metrics, documentsData, historyData] =
            await Promise.all([
                api("/api/metrics"),
                api("/api/documents"),
                api("/api/questions?limit=5")
            ]);

        state.documents = getActiveDocuments(
            documentsData.documents || []
        );

        $("statDocuments").textContent =
            formatNumber(state.documents.length);

        $("statPages").textContent =
            formatNumber(
                state.documents.reduce(
                    (total, doc) => total + Number(doc.page_count || 0),
                    0
                )
            );

        $("statChunks").textContent =
            formatNumber(
                state.documents.reduce(
                    (total, doc) => total + Number(doc.chunk_count || 0),
                    0
                )
            );

        $("statQuestions").textContent =
            formatNumber(metrics.question_count);

        renderRecentDocuments(state.documents.slice(0, 5));

        renderRecentQuestions(historyData.history || []);

    } catch (error) {
        showNotice(
            `Could not load overview: ${error.message}`,
            "error"
        );
    }
}


function renderRecentDocuments(documents) {
    const container = $("recentDocuments");

    if (!documents.length) {
        container.textContent = "No documents uploaded yet.";
        return;
    }

    container.innerHTML = documents.map((doc) => `
        <div class="history-item">
            <h4>${escapeHTML(doc.filename)}</h4>
            <p>
                ${formatNumber(doc.page_count)} pages ·
                ${formatNumber(doc.chunk_count)} chunks
            </p>
        </div>
    `).join("");
}


function renderRecentQuestions(history) {
    const container = $("recentQuestions");

    if (!history.length) {
        container.textContent = "No questions asked yet.";
        return;
    }

    container.innerHTML = history.map((item) => `
        <div class="history-item">
            <h4>${escapeHTML(item.question)}</h4>
            <p>
                ${escapeHTML(item.model || "Model unavailable")} ·
                ${formatNumber(item.processing_time_ms)} ms
            </p>
        </div>
    `).join("");
}


// --------------------------------------------------
// DOCUMENTS
// --------------------------------------------------

async function refreshDocuments() {
    try {
        const data = await api("/api/documents");

        state.documents = getActiveDocuments(data.documents || []);

        renderDocumentsTable(state.documents);

        $("documentCountLabel").textContent =
            `${state.documents.length} document(s)`;

        populateDocumentSelect();

    } catch (error) {
        showNotice(
            `Could not load documents: ${error.message}`,
            "error"
        );
    }
}


function renderDocumentsTable(documents) {
    const table = $("documentsTable");

    if (!documents.length) {
        table.innerHTML = `
            <tr>
                <td colspan="6" class="empty-state">
                    No documents available.
                </td>
            </tr>
        `;
        return;
    }

    table.innerHTML = documents.map((doc) => `
        <tr>
            <td>${escapeHTML(doc.filename)}</td>
            <td>${formatNumber(doc.page_count)}</td>
            <td>${formatNumber(doc.chunk_count)}</td>
            <td>${escapeHTML(doc.embedding?.model || "—")}</td>
            <td>
                <span class="status-pill success">
                    ${escapeHTML(doc.status || "indexed")}
                </span>
            </td>
            <td>
                <button
                    class="table-action"
                    data-document-id="${escapeHTML(doc.document_id)}"
                >
                    Inspect
                </button>
            </td>
        </tr>
    `).join("");

    table.querySelectorAll("[data-document-id]").forEach((button) => {
        button.addEventListener("click", () => {
            inspectDocument(button.dataset.documentId);
        });
    });
}


$("chooseFileButton").addEventListener("click", () => {
    $("pdfFile").click();
});


$("pdfFile").addEventListener("change", () => {
    const file = $("pdfFile").files[0];

    $("selectedFile").textContent = file
        ? `${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`
        : "No file selected";
});


$("uploadArea").addEventListener("dragover", (event) => {
    event.preventDefault();
    $("uploadArea").style.borderColor = "#f25b24";
});


$("uploadArea").addEventListener("dragleave", () => {
    $("uploadArea").style.borderColor = "";
});


$("uploadArea").addEventListener("drop", (event) => {
    event.preventDefault();

    $("uploadArea").style.borderColor = "";

    const file = event.dataTransfer.files[0];

    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".pdf")) {
        showNotice("Please select a PDF file.", "error");
        return;
    }

    const transfer = new DataTransfer();

    transfer.items.add(file);

    $("pdfFile").files = transfer.files;

    $("selectedFile").textContent =
        `${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;

    $("uploadButton").disabled = false;
});


$("pdfFile").addEventListener("change", hideNotice);


$("chooseFileButton").insertAdjacentHTML(
    "afterend",
    '<button class="button primary" id="uploadButton" style="margin-left:8px" disabled>Index PDF</button>'
);


$("pdfFile").addEventListener("change", () => {
    $("uploadButton").disabled = !$("pdfFile").files.length;
});


$("uploadButton").addEventListener("click", async () => {
    const file = $("pdfFile").files[0];

    if (!file) {
        showNotice("Choose a PDF file first.", "error");
        return;
    }

    const formData = new FormData();

    formData.append("file", file);

    setLoading("uploadProgress", true);

    $("uploadButton").disabled = true;
    $("chooseFileButton").disabled = true;

    hideNotice();

    try {
        const result = await api("/api/documents", {
            method: "POST",
            body: formData
        });

        showNotice(
            `Document indexed successfully. ${result.page_count} pages and ${result.chunk_count} chunks processed.`,
            "success"
        );

        $("pdfFile").value = "";
        $("selectedFile").textContent = "No file selected";

        await refreshDocuments();

        $("uploadButton").disabled = true;

    } catch (error) {
        showNotice(
            `Document upload failed: ${error.message}`,
            "error"
        );

        $("uploadButton").disabled = false;

    } finally {
        setLoading("uploadProgress", false);
        $("chooseFileButton").disabled = false;
    }
});


async function inspectDocument(documentId) {
    try {
        const [document, chunkData, indexStatus] =
            await Promise.all([
                api(`/api/documents/${encodeURIComponent(documentId)}`),
                api(`/api/documents/${encodeURIComponent(documentId)}/chunks`),
                api(`/api/documents/${encodeURIComponent(documentId)}/index`)
            ]);

        $("documentDetailPanel").classList.remove("hidden");

        $("documentDetailTitle").textContent =
            document.filename;

        const chunks = chunkData.chunks || [];

        $("documentDetailContent").innerHTML = `
            <p><strong>Document ID:</strong> ${escapeHTML(document.document_id)}</p>
            <p><strong>Pages:</strong> ${formatNumber(document.page_count)}</p>
            <p><strong>Chunks:</strong> ${formatNumber(document.chunk_count)}</p>
            <p><strong>Indexed chunks:</strong> ${formatNumber(indexStatus.indexed_chunks)}</p>
            <p><strong>Embedding:</strong> ${escapeHTML(document.embedding?.model || "—")}</p>
            <p><strong>Index status:</strong>
                ${indexStatus.complete ? "Complete" : "Incomplete"}
            </p>
            <div class="panel-divider"></div>
            <h3>Chunk previews</h3>
            ${chunks.map((chunk) => `
                <div class="chunk-preview">
                    <strong>${escapeHTML(chunk.chunk_id)}</strong>
                    <p>Page ${formatNumber(chunk.page)} · ${formatNumber(chunk.token_count_estimate)} estimated tokens</p>
                    <p>${escapeHTML(chunk.text.slice(0, 1000))}${chunk.text.length > 1000 ? "..." : ""}</p>
                </div>
            `).join("")}
        `;

        $("documentDetailPanel").scrollIntoView({
            behavior: "smooth"
        });

    } catch (error) {
        showNotice(
            `Could not inspect document: ${error.message}`,
            "error"
        );
    }
}


$("closeDocumentDetail").addEventListener("click", () => {
    $("documentDetailPanel").classList.add("hidden");
});


// --------------------------------------------------
// QUESTION ANSWERING
// --------------------------------------------------

function getActiveDocuments(documents) {
    return documents.filter(
        (doc) =>
            doc.status === "indexed" &&
            doc.is_active !== false
    );
}


async function populateDocumentSelect() {
    try {
        const data = await api("/api/documents");

        state.documents = getActiveDocuments(data.documents || []);

        const select = $("questionDocument");
        const previousValue = select.value;

        select.innerHTML = `
            <option value="">Select an indexed document</option>
            ${state.documents.map((doc) => `
                <option value="${escapeHTML(doc.document_id)}">
                    ${escapeHTML(doc.filename)}
                </option>
            `).join("")}
        `;

        if (state.documents.some(
            (doc) => doc.document_id === previousValue
        )) {
            select.value = previousValue;
        } else if (state.documents.length > 0) {
            select.value = state.documents[0].document_id;
        }

    } catch (error) {
        showNotice(
            `Could not load document choices: ${error.message}`,
            "error"
        );
    }
}


$("askButton").addEventListener("click", async () => {
    const documentId = $("questionDocument").value;
    const question = $("questionInput").value.trim();
    const topK = Number($("topK").value);

    if (!documentId) {
        showNotice("Please select a document first.", "error");
        return;
    }

    if (question.length < 3) {
        showNotice(
            "Enter a question with at least 3 characters.",
            "error"
        );
        return;
    }

    hideNotice();

    $("answerPanel").classList.add("hidden");

    setLoading("answerLoading", true);
    setButtonLoading("askButton", true, "Working...");

    const requestBody = {
        document_id: documentId,
        question: question,
        top_k: topK
    };

    try {
        const result = await api("/api/questions", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(requestBody)
        });

        state.lastAnswer = result;

        renderAnswer(result);

        $("answerPanel").classList.remove("hidden");

        await refreshOverview();

    } catch (error) {
        showNotice(
            `Question failed: ${error.message}`,
            "error"
        );

    } finally {
        setLoading("answerLoading", false);
        setButtonLoading("askButton", false);
    }
});


function renderAnswer(result) {
    $("answerText").textContent =
        result.answer || "No answer returned.";

    $("answerStatus").textContent =
        result.status || "Completed";

    $("answerMetadata").innerHTML = `
        <span>Model: ${escapeHTML(result.model || "—")}</span>
        <span>Processing: ${formatNumber(result.processing_time_ms)} ms</span>
        <span>Request: ${escapeHTML(result.request_id || "—")}</span>
    `;

    const sources = result.sources || [];

    $("sourceCount").textContent =
        `${sources.length} passage(s)`;

    $("evidenceList").innerHTML = sources.length
        ? sources.map((source, index) => `
            <div class="evidence-card">
                <div class="evidence-header">
                    <strong>
                        Evidence ${index + 1} · Page ${formatNumber(source.page)}
                    </strong>
                    <span class="status-pill">
                        ${escapeHTML(source.chunk_id || "—")}
                    </span>
                </div>
                <div class="evidence-content">
                    ${escapeHTML(source.text || "")}
                    <div class="evidence-scores">
                        Dense: ${source.dense_score ?? "—"} ·
                        BM25: ${source.bm25_score ?? "—"} ·
                        RRF: ${source.rrf_score ?? "—"} ·
                        Rerank: ${source.rerank_score ?? "—"}
                        <br>
                        Methods: ${escapeHTML(
                            (source.retrieval_methods || []).join(", ") || "—"
                        )}
                    </div>
                </div>
            </div>
        `).join("")
        : "<p>No evidence passages were returned.</p>";
}


// --------------------------------------------------
// QUESTION HISTORY
// --------------------------------------------------

async function refreshHistory() {
    try {
        const data = await api("/api/questions?limit=100");

        const history = data.history || [];

        const container = $("historyList");

        if (!history.length) {
            container.textContent = "No question history available.";
            return;
        }

        container.innerHTML = history.map((item) => `
            <div class="history-item">
                <h4>${escapeHTML(item.question)}</h4>
                <p>${escapeHTML(item.answer || "")}</p>
                <p>
                    ${escapeHTML(item.model || "—")} ·
                    ${formatNumber(item.processing_time_ms)} ms ·
                    ${formatDate(item.timestamp)}
                </p>
                <p>
                    ${formatNumber((item.sources || []).length)} evidence passages
                </p>
                <button
                    class="text-button"
                    data-history-id="${escapeHTML(item.request_id)}"
                >
                    View answer
                </button>
            </div>
        `).join("");

        container.querySelectorAll("[data-history-id]").forEach((button) => {
            button.addEventListener("click", async () => {
                try {
                    const result = await api(
                        `/api/questions/${encodeURIComponent(button.dataset.historyId)}`
                    );

                    showView("ask");

                    renderAnswer(result);

                    $("answerPanel").classList.remove("hidden");

                    $("questionDocument").value =
                        result.document_id;

                    $("questionInput").value =
                        result.question;

                } catch (error) {
                    showNotice(error.message, "error");
                }
            });
        });

    } catch (error) {
        showNotice(
            `Could not load history: ${error.message}`,
            "error"
        );
    }
}


$("refreshHistory").addEventListener(
    "click",
    refreshHistory
);


// --------------------------------------------------
// EVALUATION
// --------------------------------------------------

function initializeEvaluationDates() {
    const today = new Date();

    const localDate = [
        today.getFullYear(),
        String(today.getMonth() + 1).padStart(2, "0"),
        String(today.getDate()).padStart(2, "0")
    ].join("-");

    if (!$("evaluationStartDate").value) {
        $("evaluationStartDate").value = localDate;
    }

    if (!$("evaluationEndDate").value) {
        $("evaluationEndDate").value = localDate;
    }
}


function getEvaluationMetricValue(summary, metricName) {
    const value = summary?.[metricName];

    if (value === null || value === undefined) {
        return "—";
    }

    if (typeof value === "number") {
        return Number(value.toFixed(4)).toString();
    }

    return escapeHTML(value);
}


function renderEvaluationResult(result) {
    const summary = result.summary || result.metrics || {};

    const metrics = [
        ["Precision@K", summary.precision_at_k],
        ["Recall@K", summary.recall_at_k],
        ["MRR", summary.mrr],
        ["nDCG@K", summary.ndcg_at_k]
    ];

    const evaluatedCases =
        result.evaluated_cases ??
        result.evaluated_questions ??
        result.labelled_cases ??
        0;

    const unlabelledCases =
        result.unlabelled_cases ??
        result.unlabeled_cases ??
        0;

    const totalCases =
        result.total_cases ??
        result.total_questions ??
        result.processed_cases ??
        0;

    $("evaluationResult").innerHTML = `
        <div class="panel-divider"></div>

        <p>
            <strong>Evaluation run:</strong>
            ${escapeHTML(result.run_id || result.evaluation_id || "—")}
        </p>

        <p>
            <strong>Date range:</strong>
            ${escapeHTML(result.start_date || $("evaluationStartDate").value)}
            to
            ${escapeHTML(result.end_date || $("evaluationEndDate").value)}
        </p>

        <p class="muted">
            ${formatNumber(evaluatedCases)} evaluated ·
            ${formatNumber(unlabelledCases)} unlabelled ·
            ${formatNumber(totalCases)} total questions
        </p>

        <div class="metric-grid">
            ${metrics.map(([label, value]) => `
                <div class="metric-card">
                    <span>${label}</span>
                    <strong>
                        ${value === null || value === undefined
                            ? "—"
                            : getEvaluationMetricValue(summary, label === "Precision@K"
                                ? "precision_at_k"
                                : label === "Recall@K"
                                    ? "recall_at_k"
                                    : label === "MRR"
                                        ? "mrr"
                                        : "ndcg_at_k")}
                    </strong>
                </div>
            `).join("")}
        </div>

        <p class="muted">
            Recall may be unavailable when relevance judgments cover only
            retrieved top-K passages rather than the full relevant corpus.
        </p>
    `;
}


$("runEvaluationButton").addEventListener("click", async () => {
    const startDate = $("evaluationStartDate").value;
    const endDate = $("evaluationEndDate").value;
    const limit = Number($("evaluationLimit").value);
    const k = Number($("evaluationK").value);

    if (!startDate || !endDate) {
        showNotice(
            "Please select both the start date and end date.",
            "error"
        );
        return;
    }

    if (startDate > endDate) {
        showNotice(
            "Start date cannot be after end date.",
            "error"
        );
        return;
    }

    const requestBody = {
        start_date: startDate,
        end_date: endDate,
        limit: limit,
        k: k
    };

    hideNotice();

    $("evaluationResult").innerHTML = "";

    setLoading("evaluationLoading", true);

    setButtonLoading(
        "runEvaluationButton",
        true,
        "Evaluating..."
    );

    try {
        const result = await api("/api/evaluations/auto-run", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(requestBody)
        });

        renderEvaluationResult(result);

        await refreshEvaluations();

        showNotice(
            "Evaluation request completed. Review the report below.",
            "success"
        );

    } catch (error) {
        showNotice(
            `Evaluation failed: ${error.message}`,
            "error"
        );

    } finally {
        setLoading("evaluationLoading", false);

        setButtonLoading(
            "runEvaluationButton",
            false
        );
    }
});


async function refreshEvaluations() {
    try {
        const data = await api("/api/evaluations/runs");

        const runs = data.runs || [];

        $("evaluationRuns").innerHTML = runs.length
            ? runs.map((run) => `
                <div class="evaluation-item">
                    <strong>${escapeHTML(run.run_id || "—")}</strong>

                    <p>
                        ${escapeHTML(run.start_date || "—")}
                        to
                        ${escapeHTML(run.end_date || "—")}
                    </p>

                    <p>
                        ${formatNumber(run.evaluated_cases)} evaluated /
                        ${formatNumber(run.total_cases)} total questions
                    </p>

                    <p>
                        Precision@K: ${run.summary?.precision_at_k ?? "—"} ·
                        Recall@K: ${run.summary?.recall_at_k ?? "—"} ·
                        MRR: ${run.summary?.mrr ?? "—"} ·
                        nDCG@K: ${run.summary?.ndcg_at_k ?? "—"}
                    </p>
                </div>
            `).join("")
            : "No evaluation runs yet.";

    } catch (error) {
        showNotice(
            `Could not load evaluation runs: ${error.message}`,
            "error"
        );
    }
}


$("refreshEvaluations").addEventListener(
    "click",
    refreshEvaluations
);


// --------------------------------------------------
// AUDIT ACTIVITY
// --------------------------------------------------

async function refreshActivity() {
    try {
        const data = await api("/api/audit?limit=100");

        const events = data.events || [];

        $("activityList").innerHTML = events.length
            ? events.map((item) => `
                <div class="activity-item">
                    <strong>${escapeHTML(item.event)}</strong>
                    <p>${formatDate(item.timestamp * 1000)}</p>
                    <p>${escapeHTML(JSON.stringify(item.details || {}))}</p>
                </div>
            `).join("")
            : "No activity recorded yet.";

    } catch (error) {
        showNotice(
            `Could not load activity: ${error.message}`,
            "error"
        );
    }
}


$("refreshActivity").addEventListener(
    "click",
    refreshActivity
);


// --------------------------------------------------
// REFRESH
// --------------------------------------------------

$("refreshButton").addEventListener("click", async () => {
    const activeView = document.querySelector(".view.active");

    if (!activeView) return;

    const viewName = activeView.id.replace("view-", "");

    await checkHealth();

    if (viewName === "overview") {
        await refreshOverview();
    } else {
        showView(viewName);
    }
});


// --------------------------------------------------
// INITIALIZATION
// --------------------------------------------------

async function initializeApp() {
    initializeEvaluationDates();

    await checkHealth();

    await refreshOverview();
}

initializeApp();
