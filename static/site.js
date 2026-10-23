const themeButton = document.querySelector('#theme-toggle');
themeButton?.addEventListener('click', () => {
  const dark = document.documentElement.dataset.theme === 'dark';
  const theme = dark ? 'light' : 'dark';
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem('theme', theme); } catch (_) {}
});
const search = document.querySelector('#article-search');
search?.addEventListener('input', () => {
  const query = search.value.trim().toLocaleLowerCase();
  let count = 0;
  document.querySelectorAll('[data-search-entry]').forEach(entry => {
    const match = entry.textContent.toLocaleLowerCase().includes(query);
    entry.hidden = !match;
    if (match) count++;
  });
  document.querySelector('#result-count').textContent = `${count} ${count === 1 ? 'entry' : 'entries'}`;
  document.querySelector('#no-results').hidden = count !== 0;
});
