#!/usr/bin/env node

/**
 * Migration Status Dashboard
 * Displays current migration progress and statistics
 */

import { readFileSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

function loadProgress() {
  try {
    const progressPath = resolve(__dirname, '../../migration/progress.json');
    const content = readFileSync(progressPath, 'utf-8');
    return JSON.parse(content);
  } catch (error) {
    console.error('Failed to load migration progress:', error.message);
    process.exit(1);
  }
}

function getStatusEmoji(status) {
  const emojis = {
    'complete': '✅',
    'testing': '🧪',
    'in-progress': '🔄',
    'pending': '⏳',
    'blocked': '🚫',
  };
  return emojis[status] || '❓';
}

function getPriorityColor(priority) {
  const colors = {
    'critical': '\x1b[31m', // Red
    'high': '\x1b[33m',     // Yellow
    'medium': '\x1b[36m',   // Cyan
    'low': '\x1b[37m',      // White
  };
  return colors[priority] || '\x1b[0m';
}

function formatDate(dateString) {
  if (!dateString) return 'N/A';
  return new Date(dateString).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });
}

function displayDashboard(progress) {
  const reset = '\x1b[0m';
  const bold = '\x1b[1m';
  const dim = '\x1b[2m';

  console.log('\n');
  console.log(`${bold}╔════════════════════════════════════════════════════════════════╗${reset}`);
  console.log(`${bold}║         PLATINUM-PALLADIUM AI PRINTING TOOL MIGRATION         ║${reset}`);
  console.log(`${bold}╚════════════════════════════════════════════════════════════════╝${reset}`);
  console.log('');

  // Summary
  const { summary, meta } = progress;
  console.log(`${bold}📊 Migration Summary${reset}`);
  console.log(`   Phase: ${meta.phase}`);
  console.log(`   Started: ${formatDate(meta.createdAt)}`);
  console.log(`   Updated: ${formatDate(meta.updatedAt)}`);
  console.log('');

  // Progress bar
  const percentage = Math.round((summary.completed / summary.totalComponents) * 100);
  const barWidth = 40;
  const filledWidth = Math.round((percentage / 100) * barWidth);
  const emptyWidth = barWidth - filledWidth;
  const progressBar = '█'.repeat(filledWidth) + '░'.repeat(emptyWidth);

  console.log(`${bold}📈 Progress${reset}`);
  console.log(`   [${progressBar}] ${percentage}%`);
  console.log(`   Components: ${summary.completed}/${summary.totalComponents} completed`);
  console.log(`   In Progress: ${summary.inProgress}`);
  console.log(`   Pending: ${summary.pending}`);
  console.log(`   Blocked: ${summary.blocked}`);
  console.log(`   Test Coverage: ${summary.currentCoverage}% (Target: ${meta.targetCoverage}%)`);
  console.log('');

  // Component status table
  console.log(`${bold}📦 Components${reset}`);
  console.log(`   ${'Name'.padEnd(25)} ${'Status'.padEnd(12)} ${'Priority'.padEnd(10)} Coverage`);
  console.log(`   ${'-'.repeat(25)} ${'-'.repeat(12)} ${'-'.repeat(10)} ${'-'.repeat(8)}`);

  for (const component of progress.components) {
    const emoji = getStatusEmoji(component.status);
    const color = getPriorityColor(component.priority);
    const name = component.name.substring(0, 23).padEnd(25);
    const status = `${emoji} ${component.status}`.padEnd(12);
    const priority = `${color}${component.priority}${reset}`.padEnd(20); // Extra padding for color codes
    const coverage = `${component.testCoverage}%`;

    console.log(`   ${name} ${status} ${priority} ${coverage}`);
  }

  console.log('');

  // Blockers
  if (progress.blockers && progress.blockers.length > 0) {
    console.log(`${bold}🚫 Blockers${reset}`);
    for (const blocker of progress.blockers) {
      console.log(`   - ${blocker.description}`);
    }
    console.log('');
  }

  // Recent activity
  if (progress.changelog && progress.changelog.length > 0) {
    console.log(`${bold}📝 Recent Activity${reset}`);
    const recentChanges = progress.changelog.slice(-5);
    for (const change of recentChanges) {
      console.log(`   ${dim}${change.date}${reset} - ${change.action}: ${change.details}`);
    }
    console.log('');
  }

  // Next steps
  const pendingComponents = progress.components
    .filter(c => c.status === 'pending')
    .sort((a, b) => {
      const priorityOrder = { critical: 0, high: 1, medium: 2, low: 3 };
      return priorityOrder[a.priority] - priorityOrder[b.priority];
    })
    .slice(0, 3);

  if (pendingComponents.length > 0) {
    console.log(`${bold}🎯 Next Steps${reset}`);
    for (const component of pendingComponents) {
      const color = getPriorityColor(component.priority);
      console.log(`   - ${color}[${component.priority}]${reset} Migrate ${component.name}`);
    }
    console.log('');
  }

  console.log(`${dim}Run 'pnpm migrate:verify' to run equivalence tests${reset}`);
  console.log('');
}

// Main execution
const progress = loadProgress();
displayDashboard(progress);
