import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from transformers import GPT2Model, GPT2Config


class GPT4RecommendationBaseModel(nn.Module):
    '''
        The base class for collaborative GPT model, i.e.,
        the GPT model with extra user/item embeddings
    '''

    def __init__(self, config, gpt2model, a_centers, a_labels):
        super(GPT4RecommendationBaseModel, self).__init__()
        # Obtain the number of users, items
        # and the size of the original vocabulary
        self.num_users = config.num_users
        self.num_items = config.num_items
        self.vocab_size = config.vocab_size
        self.config = config

        # Create new token embeddings for user/item tokens
        self.user_embeddings = nn.Embedding(self.num_users, config.n_embd)
        self.item_embeddings = nn.Embedding(self.num_items, config.n_embd)

        # Randomly initialize the new token embeddings
        self.user_embeddings.weight.data.normal_(mean=0.0, std=config.initializer_range)
        self.item_embeddings.weight.data.normal_(mean=0.0, std=config.initializer_range)

        # The pretrained gpt2 model
        self.gpt2model = gpt2model

        self.a_centers = a_centers
        self.a_labels = a_labels

    def embed(self, input_ids):
        # input_ids is a tensor of shape (batch_size, seq_length)
        vocab_mask = (input_ids < self.vocab_size).long()
        #print('vocab_mask: ', vocab_mask)
        user_mask = ((input_ids >= self.vocab_size) & (input_ids < self.vocab_size + self.num_users)).long()
        #print('user_mask: ', vocab_mask)
        item_mask = (input_ids >= self.vocab_size + self.num_users).long()
        #print('item_mask: ', vocab_mask)

        # IDs outside of vocab range are set to 0
        vocab_ids = (input_ids * vocab_mask).clamp_(0, self.vocab_size - 1)
        vocab_embeddings = self.gpt2model.wte(vocab_ids)
        vocab_embeddings = vocab_embeddings * vocab_mask.unsqueeze(-1)

        # IDs outside of user range are set to 0
        user_ids = ((input_ids - self.vocab_size) * user_mask).clamp_(0, self.num_users - 1)
        user_embeddings = self.user_embeddings(user_ids)
        user_embeddings = user_embeddings * user_mask.unsqueeze(-1)

        # IDs outside of item range are set to 0
        item_ids = ((input_ids - self.vocab_size - self.num_users) * item_mask).clamp_(0, self.num_items - 1)
        item_embeddings = self.item_embeddings(item_ids)
        item_embeddings = item_embeddings * item_mask.unsqueeze(-1)

        # Sum up the embeddings as the input embeddings
        input_embeddings = vocab_embeddings + user_embeddings + item_embeddings

        # get node_list and a_node
        # node_list: to indicate whether the token is node or not
        node_list = 1 - vocab_mask
        # a_node: nodes  after pooling
        a_node = None
        # print('batch_size', input_ids.shape[0])
        for batch in range(input_ids.shape[0]):  # input_ids is a tensor of shape (batch_size, seq_length)
            current_seq = input_ids[batch] - self.vocab_size  # current_seq shape: (seq_length)
            user_item_mask = current_seq >= 0  # user_item_mask shape: (seq_length)
            filtered_ids = current_seq[user_item_mask]  # filtered_ids shape: (seq_length_filtered)
            current_a_node = None
            for index in range(2):
                labels = self.a_labels[index]  # labels shape: (self.num_users + self.num_items)
                centers = self.a_centers[index]  # centers shape: (10**(index+1), hidden_dim)
                # print('labels.type', labels.type)
                unique_labels = torch.unique(labels[filtered_ids])  # unique_labels shape: (seq_length_unique)
                # print('len(unique_labels)', len(unique_labels))
                # print('unique_labels', unique_labels)
                if len(unique_labels) == 0:
                    # unique_labels = [0]
                    new_centers = torch.zeros(1, centers.shape[1])  # new_centers shape: (seq_length_unique, hidden_dim)
                else:
                    new_centers = centers[unique_labels]  # new_centers shape: (seq_length_unique, hidden_dim)
                if current_a_node is None:
                    current_a_node = new_centers  # 直接设置a_node
                else:
                    current_a_node = torch.cat((current_a_node, new_centers), dim=0)  # current_a_node shape: (sum of 3 seq_length_unique, hidden_dim)
            # current_a_node_unsqueezed = current_a_node.unsqueeze(0)  # current_a_node shape: (1, sum of 3 seq_length_unique, hidden_dim)

            if a_node is None:
                a_node = [current_a_node]  # 直接设置a_node
            else:
                # print('a_node.shape()', a_node.size())
                # print('current_a_node_unsqueezed.shape()', current_a_node_unsqueezed.shape)
                # a_node = torch.cat((a_node, current_a_node_unsqueezed), dim=0)  # a_node shape: (batch_size, sum of 3 seq_length_unique, hidden_dim)
                a_node.append(current_a_node)
        a_node_padded = pad_sequence(a_node, batch_first=True)
        # print('a_node_padded.shape()', a_node_padded.shape)
        # print('input_ids_size', input_ids.shape)
        a_node = a_node_padded
        # print('a_node_padded.type()', a_node_padded.type)

        return input_embeddings, node_list, a_node
    # =============================================================================================================================================================================

    def forward(self, input_ids=None, mapping_graph_bc=None, **kwargs):
        # Obtain the embeddings of the input id sequence
        input_embeddings, node_list, a_node = self.embed(input_ids)
        # The input_embeds will be summed up with the pos_embed
        # And then forward into the transformer to get the results
        # return self.gpt2model(inputs_embeds=input_embeddings, **kwargs)
        # =============================================================================================================================================================================
        return self.gpt2model(inputs_embeds=input_embeddings, inputs_graph_bc=mapping_graph_bc, input_node_list=node_list, input_a_node=a_node, **kwargs)
        # =============================================================================================================================================================================


class CollaborativeGPTwithItemLMHeadBatch(nn.Module):
    '''
        Collaborative filtering model to learn user/item embeddings.
    '''

    def __init__(self, config, base_model):
        super(CollaborativeGPTwithItemLMHeadBatch, self).__init__()

        # Obtain the number of users, items, and vocabulary size
        self.num_users = config.num_users
        self.num_items = config.num_items
        self.vocab_size = config.vocab_size

        # Base GPT model with extended user/item ID token embeddings
        self.base_model = base_model

        # Item recommendation head
        self.item_head = nn.Linear(config.n_embd, self.num_items, bias=False)

        # Tie the weights between the item embeddings and the item recommendation head
        self.item_head.weight = self.base_model.item_embeddings.weight

    def forward(self,
                input_ids_prompt,
                input_ids_main,
                mapping_graph_bc_prompt=None,
                mapping_graph_bc_combined=None,
                labels_main=None,
                attention_mask=None,
                regularize=False,
                lambda_V=None,
                content_embeds=None,
                **kwargs):
        # Base model forward pass for the prompt text
        outputs_prompt = self.base_model(input_ids=input_ids_prompt,
                                         mapping_graph_bc=mapping_graph_bc_prompt,
                                         return_dict=True,
                                         **kwargs)
        past_key_values = outputs_prompt.past_key_values

        # Base model forward pass for the main text with attention mask
        outputs_main = self.base_model(input_ids=input_ids_main,
                                       mapping_graph_bc=mapping_graph_bc_combined,
                                       past_key_values=past_key_values,
                                       attention_mask=attention_mask,
                                       return_dict=True)

        item_logits = self.item_head(outputs_main.last_hidden_state)
        outputs = (item_logits,) + outputs_main[1:]

        if labels_main is not None:
            # Shift so that tokens < n predict n
            shift_logits = item_logits[..., :-1, :].contiguous()
            shift_labels = labels_main[..., 1:].contiguous()
            shift_labels = shift_labels - self.vocab_size - self.num_users

            # Define the loss function
            loss_fct = CrossEntropyLoss()

            # Calculate the loss only where attention mask is one
            prompt_length = input_ids_prompt.shape[1]
            main_length = input_ids_main.shape[1]

            active_loss = attention_mask[:, prompt_length + 1:].reshape(-1) == 1
            active_logits = shift_logits.view(-1, shift_logits.size(-1))[active_loss]
            active_labels = shift_labels.view(-1)[active_loss]

            # Language modeling loss for the item sequences
            loss = loss_fct(active_logits, active_labels)

            # Mutual regularization loss
            if regularize:
                collaborative_embeds = torch.cat(
                    (self.base_model.embed(input_ids_prompt)[0],
                     self.base_model.embed(input_ids_main))[0],
                    axis=1
                )
                regularize_loss = lambda_V * torch.mean(
                    nn.MSELoss(reduction='sum')(
                        collaborative_embeds,
                        content_embeds)
                )
                loss += regularize_loss
                outputs = (loss, regularize_loss) + outputs
            else:
                outputs = (loss,) + outputs
        return outputs


class ContentGPTForUserItemWithLMHeadBatch(nn.Module):
    '''
        This class conducts language modeling to learn both
        user/item token embeddings via textual data, where
        we view the texts that include user/item ID as prompt.
        E.g.,
            inputs_ids_prompt:
              "user_1 writes the following review for item_1:"
            inputs_ids_main:
              "This item is too expensive."
        where we only calculate LM loss on the main texts.
    '''

    def __init__(self, config, base_model):
        super(ContentGPTForUserItemWithLMHeadBatch, self).__init__()
        self.base_model = base_model
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Tie weights between the output layer and the token embeddings
        self.lm_head.weight = self.base_model.gpt2model.wte.weight

    def forward(self,
                input_ids_prompt,
                input_ids_main,
                mapping_graph_bc_prompt=None,
                mapping_graph_bc_combined=None,
                labels_main=None,
                attention_mask=None,
                regularize=False,
                lambda_V=None,
                collaborative_embeds=None,
                **kwargs):
        outputs_prompt = self.base_model(input_ids=input_ids_prompt,
                                         mapping_graph_bc=mapping_graph_bc_prompt,
                                         return_dict=True, **kwargs)
        past_key_values = outputs_prompt.past_key_values
        # print('========================================outputs_prompt==================================================')
        # Calculate the language modeling loss for the main texts
        outputs_main = self.base_model(input_ids=input_ids_main,
                                       mapping_graph_bc=mapping_graph_bc_combined,
                                       past_key_values=past_key_values,
                                       attention_mask=attention_mask,
                                       return_dict=True)
        # print('========================================outputs_main==================================================')

        lm_logits = self.lm_head(outputs_main.last_hidden_state)
        outputs = (lm_logits,) + outputs_main[1:]

        if labels_main is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels_main[..., 1:].contiguous()

            # Define the loss function
            loss_fct = CrossEntropyLoss()

            # Calculate the loss only where attention mask is one
            prompt_length = input_ids_prompt.shape[1]
            main_length = input_ids_main.shape[1]

            active_loss = attention_mask[:, prompt_length + 1:].reshape(-1) == 1
            active_logits = shift_logits.view(-1, shift_logits.size(-1))[active_loss]
            active_labels = shift_labels.view(-1)[active_loss]

            # Language modeling loss for the token sequences
            loss = loss_fct(active_logits, active_labels)

            # Mutual regularization loss
            if regularize:
                # User/Item token embeddings only appear in the prompt
                content_embeds = self.base_model.embed(input_ids_prompt)[0]
                regularize_loss = lambda_V * torch.mean(
                    nn.MSELoss(reduction='sum')(
                        content_embeds,
                        collaborative_embeds)
                )
                loss += regularize_loss
                outputs = (loss, regularize_loss) + outputs
            else:
                outputs = (loss,) + outputs
        return outputs


class CollaborativeGPTwithItemRecommendHead(nn.Module):
    '''
        Recommend items to a user according to input queries.
        multinomial likelihood is put on all the items for a user.
    '''

    def __init__(self, config, base_model):
        super(CollaborativeGPTwithItemRecommendHead, self).__init__()
        # Obtain the number of users and items
        self.num_users = config.num_users
        self.num_items = config.num_items

        # Base GPT model with extended user/item ID token embeddings
        self.base_model = base_model

        # Item recommendation head
        self.item_head = nn.Linear(config.n_embd, self.num_items, bias=False)

        # Tie the weights between the item embeddings and the item recommendation head
        self.item_head.weight = self.base_model.item_embeddings.weight

    def forward(self,
                input_ids=None,
                target_ids=None,
                # target_ids_neg=None,
                mapping_graph_bc=None,
                attention_mask=None,
                regularize=False,
                lambda_V=None,
                main_ids=None,
                content_embeds=None,
                **kwargs):
        transformer_outputs = self.base_model(input_ids,
                                              mapping_graph_bc=mapping_graph_bc,
                                              attention_mask=attention_mask,
                                              **kwargs)
        # print('input_ids: ', input_ids)
        hidden_states = transformer_outputs[0]
        # print('hidden_states: ', hidden_states)

        # Find the indices of the last non-padding tokens
        last_non_pad_token_indices = attention_mask.sum(dim=1) - 1
        # print('the indices of the last non-padding tokens: ', last_non_pad_token_indices)

        # Gather the last non-padding token embeddings
        last_token_hidden_states = torch.stack([
            hidden_states[i, idx, :] for i, idx in \
            enumerate(last_non_pad_token_indices)
        ])
        # print('the last non-padding token embeddings: ', last_token_hidden_states)

        # Calculate the item scores
        item_scores = self.item_head(last_token_hidden_states)
        # print('item scores: ', item_scores)

        # # strong rules for generation
        # if target_ids_neg is not None:
        #     for ids in target_ids_neg:
        #         item_scores[ids] = 0

        # Convert scores to multinomial probabilities
        item_log_probs = F.log_softmax(item_scores, dim=-1)
        # print('multinomial probabilities: ', item_log_probs)

        # Calculating the multinomial loss
        neg_ll = -torch.mean(torch.sum(item_log_probs * target_ids, dim=-1))
        # print('multinomial loss: ', neg_ll)

        if regularize:
            # User/Item token embeddings only appear in the prompt
            rec_embeds_prompt = self.base_model.embed(input_ids)[0]
            rec_embeds_target = self.base_model.embed(main_ids)[0]
            rec_embeds = torch.cat(
                (rec_embeds_prompt, rec_embeds_target),
                axis=1
            )
            regularize_loss = lambda_V * torch.mean(
                nn.MSELoss(reduction='sum')(
                    rec_embeds,
                    content_embeds)
            )
            neg_ll += regularize_loss
            outputs = (neg_ll, regularize_loss, item_log_probs)
        else:
            outputs = (neg_ll, item_log_probs)
        return outputs
