SELECT
    ftr.result_id,
    ftr.created_at,
    ftr.test_accuracy,
    ftr.valid_accuracy,
    ftr.feature_extract_time,
    ftr.training_time,
    dm.model_name,
    dp.batch_size,
    dp.num_workers,
    dp.sampling_method,
    dh.objective,
    dh.eval_metric,
    dh.n_estimators,
    dh.max_depth,
    dd.train_samples,
    dd.valid_samples,
    dd.test_samples
FROM fact_training_results AS ftr
LEFT JOIN dim_model AS dm ON ftr.model_id = dm.model_id
LEFT JOIN dim_preprocessing AS dp ON ftr.preprocessing_id = dp.preprocessing_id
LEFT JOIN dim_hyperparameters AS dh ON ftr.hyperparameter_id = dh.hyperparameter_id
LEFT JOIN dim_dataset AS dd ON ftr.dataset_id = dd.dataset_id
ORDER BY ftr.test_accuracy DESC, ftr.feature_extract_time ASC, ftr.training_time ASC
LIMIT 1;