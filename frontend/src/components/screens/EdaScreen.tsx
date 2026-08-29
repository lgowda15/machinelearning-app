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

/** Screen 2 (frontend.md): two columns -- summary statistics table left,
 * distribution charts right with the target first -- plus the dismissible
 * imbalance note, shown only here. */
export function EdaScreen({ profile }: EdaScreenProps) {
  const [imbalanceVisible, setImbalanceVisible] = useState(true);

  const orderedColumns = useMemo(() => {
    const target = profile.columns.filter((c) => c.is_target);
    const rest = profile.columns.filter((c) => !c.is_target);
    return [...target, ...rest];
  }, [profile.columns]);

  // The target column is the "thing being predicted" -- it gets the colour
  // of the model type its data_type will train, per frontend.md's colour
  // rule; every other distribution stays --ink. No model has been chosen
  // yet at this screen, so this is keyed off data_type, not model_type.
  const targetAccentVar = targetColorVarForDataType(profile.data_type);

  return (
    <ScreenPanel maxWidthClassName={WORKSPACE_WIDTH}>
      {profile.class_imbalance && imbalanceVisible && (
        <ImbalanceBanner info={profile.class_imbalance} onDismiss={() => setImbalanceVisible(false)} />
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        <section>
          <h2 className="mb-2 text-sm font-medium uppercase tracking-wide text-muted">
            Summary statistics
          </h2>
          <SummaryTable columns={profile.columns} targetAccentVar={targetAccentVar} />
        </section>

        <section>
          <h2 className="mb-2 text-sm font-medium uppercase tracking-wide text-muted">Distributions</h2>
          <div className="flex flex-col gap-4">
            {orderedColumns.map((column) => (
              <DistributionChart key={column.name} column={column} targetAccentVar={targetAccentVar} />
            ))}
          </div>
        </section>
      </div>
    </ScreenPanel>
  );
}
