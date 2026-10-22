import { useContext, useEffect, useMemo, useState } from 'react';

import {
  Alert,
  ArrowSwitch,
  Book,
  Calendar,
  CheckBox,
  Download,
  File,
  FileDirectory,
  Filter,
  Note,
  Plus,
} from '@osrd-project/ui-icons';
import cx from 'classnames';
import { useTranslation } from 'react-i18next';

import { useItineraryModalContext } from 'applications/operationalStudies/hooks/useItineraryModalContext';
import { useScenarioContext } from 'applications/operationalStudies/hooks/useScenarioContext';
import useScenarioTrainScheduleSet, {
  type ImportTrainScheduleSetsPayload,
} from 'applications/operationalStudies/hooks/useScenarioTrainScheduleSet';
import { useTimetableContext } from 'applications/operationalStudies/hooks/useTimetableContext';
import { osrdEditoastApi } from 'common/api/osrdEditoastApi';
import { ModalContext } from 'common/BootstrapSNCF/ModalSNCF/ModalProvider';
import MenuTriggerButton from 'common/MenuTriggerButton';
import UploadFileModal from 'common/uploadFileModal';
import type { TrainScheduleWithDetails } from 'modules/trainSchedule/types';
import { setFailure } from 'reducers/main';
import { useAppDispatch } from 'store';
import { Duration } from 'utils/duration';
import { castErrorToFailure } from 'utils/error';

import useImportTrainSchedules from '../ImportTrainSchedule';
import { TrainScheduleSetCatalogDialog } from '../ImportTrainScheduleSets';
import RoundTripsModal from '../RoundTrips/RoundTripsModal';
import FilterPanel from './FilterPanel';
import SelectionToolBar from './TimetableSelectionToolbar';
import type { TimetableFilters, TimetableMode } from './types';
import { exportTrainSchedules, timetableHasInvalidTrainSchedule } from './utils';

type TimetableToolbarProps = {
  timetableFilters: TimetableFilters;
  trainSchedulesWithDetails: TrainScheduleWithDetails[];
  filteredTrainSchedules: TrainScheduleWithDetails[];
  selectedTrainScheduleIds: number[];
  showTrainDetails: boolean;
  isSelectMode: boolean;
  setSelectedTrainScheduleIds: (selectedTrainScheduleIds: number[]) => void;
  setShowTrainDetails: (show: boolean) => void;
  setIsSelectMode: (show: boolean) => void;
  refreshNge: () => Promise<void>;
  handleDeleteTrainSchedules: () => void;
  handleMoveTrainSchedules: () => void;
  timetableMode: TimetableMode;
  setTimetableMode: React.Dispatch<React.SetStateAction<TimetableMode>>;
};

const TimetableToolbar = ({
  timetableFilters,
  trainSchedulesWithDetails,
  filteredTrainSchedules,
  selectedTrainScheduleIds,
  showTrainDetails,
  isSelectMode,
  setShowTrainDetails,
  setIsSelectMode,
  setSelectedTrainScheduleIds,
  refreshNge,
  handleDeleteTrainSchedules,
  handleMoveTrainSchedules,
  timetableMode,
  setTimetableMode,
}: TimetableToolbarProps) => {
  const { t, i18n } = useTranslation(['operational-studies', 'translation'], { keyPrefix: 'main' });
  const dispatch = useAppDispatch();

  const { infraId, timetableId, scenario } = useScenarioContext();
  const { trainSchedules } = useTimetableContext();
  const [plannerAvailable, setPlannerAvailable] = useState(false);
  useEffect(() => {
    let active = true;
    let timeout: ReturnType<typeof setTimeout> | undefined;
    let controller: AbortController | undefined;
    const checkPlannerAvailability = async () => {
      controller = new AbortController();
      const requestTimeout = setTimeout(() => controller?.abort(), 5_000);
      try {
        const response = await fetch('/ops-intelligence/readyz', { signal: controller.signal });
        const readiness = response.ok ? await response.json() : undefined;
        if (active) {
          setPlannerAvailable(
            readiness?.status === 'ready' && readiness?.planning_configured === true
          );
        }
      } catch {
        if (active) setPlannerAvailable(false);
      } finally {
        clearTimeout(requestTimeout);
        if (active) timeout = setTimeout(checkPlannerAvailability, 15_000);
      }
    };
    void checkPlannerAvailability();
    return () => {
      active = false;
      controller?.abort();
      if (timeout) clearTimeout(timeout);
    };
  }, []);
  const selectedSchedule =
    selectedTrainScheduleIds.length === 1
      ? trainSchedules.get(selectedTrainScheduleIds[0])
      : undefined;
  const pathTemplate = selectedSchedule?.path.map(({ key, location }) => {
    const stopFor = selectedSchedule.schedule?.find((item) => item.at === key)?.stop_for;
    return {
      key,
      location,
      ...(stopFor ? { stop_for_seconds: Math.round(Duration.parse(stopFor).total('second')) } : {}),
    };
  });
  const plannerParams = new URLSearchParams({
    infra: String(infraId),
    timetable: String(timetableId),
    scenario: String(scenario.id),
    lang: i18n.resolvedLanguage || i18n.language || 'en',
    'timetable-type': scenario.timetable_type,
    ...(selectedTrainScheduleIds.length
      ? { 'eligible-schedule-ids': selectedTrainScheduleIds.join(',') }
      : {}),
    ...(selectedSchedule
      ? {
          name: selectedSchedule.train_name,
          'rolling-stock': selectedSchedule.rolling_stock_name,
        }
      : {}),
    ...(pathTemplate ? { path: JSON.stringify(pathTemplate) } : {}),
  });
  const plannerUrl = `/ops-intelligence/planner?${plannerParams}`;

  const { data: trainScheduleRoundTripsData } =
    osrdEditoastApi.endpoints.getTimetableByIdRoundTrips.useQuery({
      id: timetableId,
    });

  const [isFilterPanelOpen, setIsFilterPanelOpen] = useState(false);
  const [roundTripsModalIsOpen, setRoundTripsModalIsOpen] = useState(false);

  const { openModal, closeModal } = useContext(ModalContext);
  const importFile = useImportTrainSchedules();

  const { openItineraryModalToCreate } = useItineraryModalContext();

  const toggleisSelectMode = () => {
    setIsSelectMode(!isSelectMode);
  };

  const toggleShowTrainDetails = () => {
    setShowTrainDetails(!showTrainDetails);
  };

  const toggleFilterPanel = () => {
    setIsFilterPanelOpen(!isFilterPanelOpen);
  };

  const handleExportTrainSchedules = () => {
    exportTrainSchedules(
      selectedTrainScheduleIds,
      trainSchedules,
      trainScheduleRoundTripsData?.results
    );
  };

  const areAllTrainchedulesSelected =
    selectedTrainScheduleIds.length === filteredTrainSchedules.length;

  const areInvalidTrainSchedules = timetableHasInvalidTrainSchedule(filteredTrainSchedules);

  const toggleAllTrainsSelection = () => {
    if (!areAllTrainchedulesSelected) {
      const trainSchedulesDisplayed = filteredTrainSchedules.map(({ id }) => id);
      setSelectedTrainScheduleIds(trainSchedulesDisplayed);
    } else {
      setSelectedTrainScheduleIds([]);
    }
  };

  const openTimetableImportModal = () =>
    openModal(
      <UploadFileModal
        handleSubmit={async (file) => {
          closeModal();
          await importFile(file);
        }}
      />
    );

  const { importTrainScheduleSets } = useScenarioTrainScheduleSet(trainSchedulesWithDetails);
  const [trainScheduleSetsCatalogModalIsOpen, setTrainScheduleSetsCatalogModalIsOpen] =
    useState<boolean>(false);

  const handleImportTrainScheduleSets = async (data: ImportTrainScheduleSetsPayload) => {
    try {
      await importTrainScheduleSets(data);
      setTrainScheduleSetsCatalogModalIsOpen(false);
    } catch (error) {
      dispatch(setFailure(castErrorToFailure(error)));
    }
  };

  const trainSchedulesList = useMemo(() => [...trainSchedules.values()], [trainSchedules]);

  return (
    <>
      {areInvalidTrainSchedules && (
        <div className="invalid-trains">
          <Alert size="sm" variant="fill" />
          <span data-testid="invalid-train-schedule-message" className="invalid-trains-message">
            {t('timetable.invalidTrains')}
          </span>
        </div>
      )}
      <div
        className={cx('timetable-toolbar', {
          'are-invalid-items': areInvalidTrainSchedules,
        })}
      >
        <div className="left-group">
          <button
            className={cx('select-options-button', {
              active: isSelectMode,
            })}
            data-testid="scenarios-select-options-button"
            title={t('timetable.selectOptions')}
            onClick={toggleisSelectMode}
            disabled={trainSchedules.size === 0}
            type="button"
          >
            <CheckBox />
          </button>
          <button
            className={cx('train-detail-button', {
              active: showTrainDetails,
            })}
            data-testid="scenarios-show-train-details-button"
            title={showTrainDetails ? t('lessDetails') : t('moreDetails')}
            onClick={toggleShowTrainDetails}
            disabled={trainSchedules.size === 0}
            type="button"
          >
            <Note />
          </button>
          <button
            className="round-trip-button"
            data-testid="scenarios-manage-round-trips-button"
            title={t('roundTripsModal.manageRoundTrips')}
            onClick={() => setRoundTripsModalIsOpen(true)}
            disabled={trainSchedules.size === 0}
            type="button"
          >
            <ArrowSwitch />
          </button>
          <MenuTriggerButton
            buttonProps={{
              icon: <Download />,
              dataTestID: 'scenarios-import-train-schedule-button',
              title: t('timetable.importTimetableNetworkGraphItem'),
              className: 'import-button',
            }}
            menuProps={{
              items: [
                {
                  icon: <File />,
                  title: t('timetable.importTimetableNetworkGraphFromFile'),
                  dataTestID: 'scenarios-import-timetable-by-file',
                  onClick: () => openTimetableImportModal(),
                },
                {
                  icon: <Book />,
                  title: t('timetable.importTimetableNetworkGraphFromCatalog'),
                  dataTestID: 'scenarios-import-timetable-by-catalog',
                  onClick: () => setTrainScheduleSetsCatalogModalIsOpen(true),
                },
              ],
            }}
          />
          <button
            className="add-button"
            data-testid="scenarios-add-train-schedule-button"
            title={t('timetable.addTrainSchedule')}
            onClick={() => {
              openItineraryModalToCreate();
            }}
            type="button"
          >
            <Plus />
          </button>
          {plannerAvailable && (
            <a
              className="add-button"
              data-testid="scenarios-plan-service-button"
              title={t('timetable.planServiceOptions')}
              aria-label={t('timetable.planServiceOptions')}
              href={plannerUrl}
              target="_blank"
              rel="noopener noreferrer"
            >
              <File />
            </a>
          )}
          {!plannerAvailable && (
            <button
              className="add-button"
              data-testid="scenarios-plan-service-button"
              title={t('timetable.planServiceUnavailable')}
              aria-label={t('timetable.planServiceUnavailable')}
              aria-disabled="true"
              disabled
              type="button"
            >
              <File />
            </button>
          )}
        </div>
        <div className="right-group">
          <button
            className={cx('filter-button', {
              active: isFilterPanelOpen,
            })}
            data-testid="timetable-filter-button"
            title={t('timetable.toggleFilters')}
            onClick={toggleFilterPanel}
            disabled={trainSchedules.size === 0}
            type="button"
          >
            <Filter />
          </button>
          <div className="timetable-sort-switch">
            <button
              className={cx('timetable-mode-button', {
                active: timetableMode === 'chronological',
              })}
              data-testid="timetable-chronological-mode-button"
              title={t('timetable.modeChronological')}
              onClick={() => setTimetableMode('chronological')}
              type="button"
            >
              <Calendar />
            </button>

            <button
              className={cx('timetable-mode-button', {
                active: timetableMode === 'trainScheduleSet',
              })}
              data-testid="timetable-train-schedule-set-mode-button"
              title={t('timetable.modeTrainScheduleSet')}
              onClick={() => setTimetableMode('trainScheduleSet')}
              type="button"
            >
              <FileDirectory />
            </button>
          </div>
        </div>
      </div>
      {isSelectMode && filteredTrainSchedules.length > 0 && (
        <SelectionToolBar
          selectedTrainScheduleIds={selectedTrainScheduleIds}
          areAllTrainchedulesSelected={areAllTrainchedulesSelected}
          areInvalidTrainSchedules={areInvalidTrainSchedules}
          toggleAllTrainsSelection={toggleAllTrainsSelection}
          handleExportTrainSchedules={handleExportTrainSchedules}
          handleDeleteTrainSchedules={handleDeleteTrainSchedules}
          handleMoveTrainSchedules={handleMoveTrainSchedules}
        />
      )}
      {isFilterPanelOpen && (
        <div
          className={cx('sticky-filter', {
            'are-invalid-items': areInvalidTrainSchedules,
            'selection-mode-open': isSelectMode,
          })}
        >
          <FilterPanel toggleFilterPanel={toggleFilterPanel} timetableFilters={timetableFilters} />
        </div>
      )}
      {roundTripsModalIsOpen && (
        <RoundTripsModal
          roundTripsModalIsOpen={roundTripsModalIsOpen}
          setRoundTripsModalIsOpen={setRoundTripsModalIsOpen}
          infraId={infraId}
          timetableId={timetableId}
          trainSchedules={trainSchedulesList}
          refreshNge={refreshNge}
        />
      )}
      {trainScheduleSetsCatalogModalIsOpen && (
        <TrainScheduleSetCatalogDialog
          onSubmit={handleImportTrainScheduleSets}
          onCancel={() => setTrainScheduleSetsCatalogModalIsOpen(false)}
        />
      )}
    </>
  );
};

export default TimetableToolbar;
