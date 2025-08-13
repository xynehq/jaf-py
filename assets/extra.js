// Custom JavaScript for JAF Python Documentation

document.addEventListener('DOMContentLoaded', function() {
    // Add copy button functionality for code blocks
    addCopyButtons();
    
    // Initialize tooltips
    initializeTooltips();
    
    // Add smooth scrolling for anchor links
    initializeSmoothScrolling();
    
    // Initialize collapsible sections
    initializeCollapsibleSections();
});

function addCopyButtons() {
    const codeBlocks = document.querySelectorAll('pre > code');
    
    codeBlocks.forEach(function(codeBlock) {
        const pre = codeBlock.parentNode;
        const button = document.createElement('button');
        button.className = 'copy-button';
        button.innerHTML = '📋 Copy';
        button.title = 'Copy to clipboard';
        
        button.addEventListener('click', function() {
            navigator.clipboard.writeText(codeBlock.textContent).then(function() {
                button.innerHTML = '✅ Copied!';
                button.style.color = '#4caf50';
                
                setTimeout(function() {
                    button.innerHTML = '📋 Copy';
                    button.style.color = '';
                }, 2000);
            });
        });
        
        pre.style.position = 'relative';
        pre.appendChild(button);
    });
}

function initializeTooltips() {
    const tooltipElements = document.querySelectorAll('[data-tooltip]');
    
    tooltipElements.forEach(function(element) {
        element.addEventListener('mouseenter', function() {
            showTooltip(element, element.getAttribute('data-tooltip'));
        });
        
        element.addEventListener('mouseleave', function() {
            hideTooltip();
        });
    });
}

function showTooltip(element, text) {
    const tooltip = document.createElement('div');
    tooltip.className = 'custom-tooltip';
    tooltip.textContent = text;
    tooltip.style.position = 'absolute';
    tooltip.style.background = '#333';
    tooltip.style.color = '#fff';
    tooltip.style.padding = '0.5rem';
    tooltip.style.borderRadius = '4px';
    tooltip.style.fontSize = '0.875rem';
    tooltip.style.zIndex = '1000';
    tooltip.style.maxWidth = '200px';
    tooltip.style.wordWrap = 'break-word';
    
    document.body.appendChild(tooltip);
    
    const rect = element.getBoundingClientRect();
    tooltip.style.left = rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2) + 'px';
    tooltip.style.top = rect.top - tooltip.offsetHeight - 10 + 'px';
}

function hideTooltip() {
    const tooltip = document.querySelector('.custom-tooltip');
    if (tooltip) {
        tooltip.remove();
    }
}

function initializeSmoothScrolling() {
    const links = document.querySelectorAll('a[href^="#"]');
    
    links.forEach(function(link) {
        link.addEventListener('click', function(e) {
            const targetId = link.getAttribute('href').substring(1);
            const targetElement = document.getElementById(targetId);
            
            if (targetElement) {
                e.preventDefault();
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
                
                // Update URL without jumping
                history.pushState(null, null, '#' + targetId);
            }
        });
    });
}

function initializeCollapsibleSections() {
    const collapsibleHeaders = document.querySelectorAll('.collapsible-header');
    
    collapsibleHeaders.forEach(function(header) {
        header.style.cursor = 'pointer';
        header.style.userSelect = 'none';
        
        header.addEventListener('click', function() {
            const content = header.nextElementSibling;
            if (content && content.classList.contains('collapsible-content')) {
                const isVisible = content.style.display !== 'none';
                content.style.display = isVisible ? 'none' : 'block';
                
                // Toggle arrow indicator
                const arrow = header.querySelector('.arrow');
                if (arrow) {
                    arrow.textContent = isVisible ? '▶' : '▼';
                }
            }
        });
    });
}

// Add CSS for copy buttons
const style = document.createElement('style');
style.textContent = `
    .copy-button {
        position: absolute;
        top: 0.5rem;
        right: 0.5rem;
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 0.25rem 0.5rem;
        font-size: 0.75rem;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    
    .copy-button:hover {
        background: rgba(255, 255, 255, 0.95);
    }
    
    .collapsible-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .collapsible-header .arrow {
        font-size: 0.8em;
        transition: transform 0.2s;
    }
    
    .collapsible-content {
        margin-left: 1rem;
        border-left: 2px solid #e0e0e0;
        padding-left: 1rem;
    }
`;

document.head.appendChild(style);