import { useMemo, useState } from "react";
import { DistributionChart } from "../DistributionChart";
import { ImbalanceBanner } from "../ImbalanceBanner";
import { ScreenPanel, WORKSPACE_WIDTH } from "../ScreenPanel";
import { SummaryTable } from "../SummaryTable";
import { targetColorVarForDataType } from "../../lib/modelType";
import type { DataProfileResponse } from "../../hooks/useDataset";

interface EdaScreenProps {
  profile: DataProfileResponse;
}

/**
 * Screen 2: EDA
 *
 * Layout:
 * 1. Summary statistics table
 * 2. Distribution charts in a responsive 3-column grid
 * 3. Target column appears first among the distributions
 * 4. Imbalance note is shown only on this screen
 */
export function EdaScreen({ profile }: EdaScreenProps) {
  const [imbalanceVisible, setImbalanceVisible] = useState(true);

  const orderedColumns = useMemo(() => {
    const target = profile.columns.filter((c) => c.is_target);
    const rest = profile.columns.filter((c) => !c.is_target);

    return [...target, ...rest];
  }, [profile.columns]);

  // The target column gets the colour associated with the detected
  // data type. Other distributions remain in the default colour.
  const targetAccentVar = targetColorVarForDataType(profile.data_type);

  return (
    <ScreenPanel maxWidthClassName={WORKSPACE_WIDTH}>
      {profile.class_imbalance && imbalanceVisible && (
        <ImbalanceBanner
          info={profile.class_imbalance}
          onDismiss={() => setImbalanceVisible(false)}
        />
      )}

      <div className="flex flex-col gap-8">
        {/* Summary Statistics */}
        <section>
          <h2 className="mb-2 text-sm font-medium uppercase tracking-wide text-muted">
            Summary statistics
          </h2>

          <SummaryTable
            columns={profile.columns}
            targetAccentVar={targetAccentVar}
          />
        </section>

        {/* Distributions */}
        <section>
          <h2 className="mb-2 text-sm font-medium uppercase tracking-wide text-muted">
            Distributions
          </h2>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
            {orderedColumns.map((column) => (
              <DistributionChart
                key={column.name}
                column={column}
                targetAccentVar={targetAccentVar}
              />
            ))}
          </div>
        </section>
      </div>
    </ScreenPanel>
  );
}