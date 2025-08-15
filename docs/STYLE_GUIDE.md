# JAF-PY Documentation Style Guide

This document outlines the professional documentation standards established for the JAF-PY project to maintain consistency, accuracy, and professionalism across all documentation.

## Visual Standards

### Typography
- **Primary Font**: Inter (body text)
- **Code Font**: JetBrains Mono (code blocks and inline code)
- **Line Height**: 1.7 for optimal readability
- **Font Size**: 16px base for accessibility

### Color Palette
- **Primary**: Indigo (`#3f51b5`) - professional, trustworthy
- **Accent**: Deep Orange (`#ff5722`) - controlled contrast
- **Success**: Dark Green (`#2e7d32`) - conservative success color
- **Warning**: Orange (`#f57c00`) - clear but not alarming
- **Error**: Dark Red (`#c62828`) - serious but not aggressive

### Layout Principles
- **Whitespace**: Generous spacing between sections (2.5rem)
- **Code Blocks**: Subtle background with clean borders
- **Hero Section**: Minimal, focused messaging
- **Feature Cards**: Clean cards with subtle hover effects

## Content Standards

### Tone of Voice
- **Professional and Direct**: No casual language or informal expressions
- **Confident**: Avoid hedge words like "pretty," "quite," "somewhat"
- **Precise**: Use specific technical terms rather than vague descriptions
- **Accessible**: Complex concepts explained clearly without jargon

### Prohibited Elements
- **No Emojis**: Zero tolerance for any emoji usage in documentation
- **No Exclamation Marks**: Professional tone doesn't require excitement
- **No Casual Expressions**: Avoid "awesome," "super easy," "cool," etc.
- **No Marketing Hyperbole**: Focus on technical accuracy over promotional language

### Code Examples
- **Runnable**: Every code example must be tested and verified to work
- **Complete**: Include all necessary imports and context
- **Safe**: No eval() or other potentially dangerous patterns
- **Type-Safe**: Include proper type hints and Pydantic models
- **Error Handling**: Show proper exception handling patterns

### Documentation Structure
- **Hierarchy**: Clear H1/H2/H3 structure with logical flow
- **Cross-References**: Use relative links between documentation pages
- **API References**: Include parameter types, return values, and examples
- **Examples Before Theory**: Show working code first, then explain

## Navigation Standards

### Information Architecture
1. **Get Started** - Immediate action path
2. **Understand Concepts** - Core principles and architecture
3. **Build with JAF** - Practical implementation guides
4. **Production Features** - Advanced capabilities
5. **Examples** - Working demonstrations
6. **Deploy** - Production deployment
7. **Reference** - API documentation and troubleshooting

### Page Naming
- **Short and Clear**: Maximum 3 words per navigation item
- **Action-Oriented**: Start with verbs where appropriate
- **Consistent**: Similar content types use similar naming patterns

## Quality Assurance

### Code Validation
- All Python examples must pass syntax checking
- Import statements must be accurate to actual codebase
- Function signatures must match implemented APIs
- Return types and error handling must be correct

### Technical Accuracy
- Claims about performance must be substantiated
- Integration examples must reference actual dependencies
- Configuration examples must use correct parameter names
- Version-specific features must be clearly marked

### Accessibility
- High contrast ratios for all text
- Descriptive link text
- Proper heading hierarchy
- Keyboard navigation support

## Maintenance

### Review Process
- All documentation changes require accuracy validation
- Code examples must be tested before publication
- Technical claims must be verified against implementation
- Style consistency must be maintained

### Version Control
- Documentation versioning aligned with code releases
- Deprecation notices for removed features
- Migration guides for breaking changes
- Change log maintenance

This style guide ensures that JAF-PY documentation maintains the highest professional standards and provides users with accurate, accessible, and actionable information.