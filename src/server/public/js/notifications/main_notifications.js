export function initNotifications() {
    // -------- Notification Element (for both) --------
    const notification = document.getElementById('notification');



}

// notifications.js
export function showNotification(message, isError = false) {
    const notification = document.createElement('div');
    notification.className = isError ? 'notification error' : 'notification success';
    notification.textContent = message;

    document.body.appendChild(notification);
    setTimeout(() => {
        notification.remove();
    }, 3000);
}