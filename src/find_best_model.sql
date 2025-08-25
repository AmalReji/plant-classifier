select * from model_results
where test_accuracy >0.85 and valid_accuracy>0.85
order by test_accuracy desc, valid_accuracy desc, training_time asc;