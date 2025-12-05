#!/bin/bash
# Monitor Trial 4 completion and report metrics

OPTUNA_DB="optuna.db"
CHECK_INTERVAL=300  # Check every 5 minutes

echo "=== Trial 4 Monitor Started ==="
echo "Started: $(date)"
echo "Checking every 5 minutes..."
echo ""

while true; do
    # Check trial status
    TRIAL_STATUS=$(python3 -c "
import sqlite3
conn = sqlite3.connect('$OPTUNA_DB')
cursor = conn.cursor()
cursor.execute('SELECT state FROM trials WHERE trial_id = 4')
result = cursor.fetchone()
conn.close()
print(result[0] if result else 'NOT_FOUND')
" 2>/dev/null)

    CURRENT_TIME=$(date '+%Y-%m-%d %H:%M:%S')

    if [ "$TRIAL_STATUS" = "COMPLETE" ]; then
        echo ""
        echo "================================================================"
        echo "🎉 TRIAL 4 COMPLETED! 🎉"
        echo "Time: $CURRENT_TIME"
        echo "================================================================"
        echo ""

        # Get the metric value
        python3 -c "
import sqlite3
conn = sqlite3.connect('$OPTUNA_DB')
cursor = conn.cursor()
cursor.execute('''
    SELECT tv.value
    FROM trials t
    LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
    WHERE t.trial_id = 4
''')
result = cursor.fetchone()
conn.close()

if result and result[0] is not None:
    print(f'✓ Trial 4 Macro-F1 Score: {result[0]:.4f}')
    print('')
    print('This is your FIRST COMPLETED TRIAL with augmentation enabled!')
else:
    print('⚠ Trial completed but no value recorded')
"

        # Show current epoch from log
        echo ""
        echo "Last training output:"
        tail -3 hpo_aug.log

        break

    elif [ "$TRIAL_STATUS" = "FAIL" ]; then
        echo ""
        echo "================================================================"
        echo "❌ TRIAL 4 FAILED"
        echo "Time: $CURRENT_TIME"
        echo "================================================================"
        echo ""
        echo "Last log output:"
        tail -20 hpo_aug.log
        break

    elif [ "$TRIAL_STATUS" = "PRUNED" ]; then
        echo ""
        echo "================================================================"
        echo "✂️ TRIAL 4 PRUNED (Early Stopped)"
        echo "Time: $CURRENT_TIME"
        echo "================================================================"
        break

    else
        # Still running - show progress
        CURRENT_EPOCH=$(tail -1 hpo_aug.log | grep -oP 'Epoch \K[0-9]+' | head -1)
        echo "[$CURRENT_TIME] Trial 4: RUNNING (Epoch $CURRENT_EPOCH/100)"
    fi

    sleep $CHECK_INTERVAL
done

echo ""
echo "Monitor finished at: $(date)"
