// Theme handling
function initializeTheme() {
    const themeToggle = document.getElementById('theme-toggle');
    const themeIcon = themeToggle.querySelector('i');
    
    // Check for saved theme preference
    const savedTheme = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeButton(savedTheme);
    
    // Theme toggle click handler
    themeToggle.addEventListener('click', () => {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        updateThemeButton(newTheme);
    });
}

function updateThemeButton(theme) {
    const themeToggle = document.getElementById('theme-toggle');
    const themeIcon = themeToggle.querySelector('i');
    
    if (theme === 'light') {
        themeIcon.className = 'bi bi-moon-fill';
        themeToggle.innerHTML = '<i class="bi bi-moon-fill"></i> Dark Mode';
    } else {
        themeIcon.className = 'bi bi-sun-fill';
        themeToggle.innerHTML = '<i class="bi bi-sun-fill"></i> Light Mode';
    }
} 