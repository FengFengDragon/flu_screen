const API_BASE = '';

async function apiCall(url, options = {}) {
    const response = await fetch(url, {
        ...options,
        headers: {
            'Content-Type': 'application/json',
            ...options.headers
        }
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || '请求失败');
    }
    return response.json();
}

const Modal = {
    show: (modalId) => {
        document.getElementById(modalId).style.display = 'block';
    },
    hide: (modalId) => {
        document.getElementById(modalId).style.display = 'none';
    }
};

function formatDate(dateStr) {
    if (!dateStr) return '-';
    const date = new Date(dateStr);
    return date.toLocaleDateString('zh-CN');
}

function formatNumber(num, decimals = 2) {
    if (num === null || num === undefined) return '-';
    return parseFloat(num).toFixed(decimals);
}

function getStatusBadge(status) {
    const badges = {
        'pending': '<span class="badge badge-pending">待处理</span>',
        'completed': '<span class="badge badge-completed">已完成</span>',
        'failed': '<span class="badge badge-failed">失败</span>'
    };
    return badges[status] || `<span class="badge">${status}</span>`;
}
