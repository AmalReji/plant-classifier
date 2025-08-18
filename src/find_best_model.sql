select * from model_results
where test_accuracy >0.8 and valid_accuracy>0.8
order by test_accuracy desc, valid_accuracy desc